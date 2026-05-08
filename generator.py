"""Grounded answer generation with citations and conversation memory."""

from __future__ import annotations

from collections import deque
import logging
import time
from typing import Deque, Dict, Iterator, List, Tuple

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency availability is environmental.
    tiktoken = None

from config import MEMORY_WINDOW, Settings, settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency availability is environmental.
    OpenAI = None


LOGGER = logging.getLogger(__name__)


class _LLMRateLimiter:
    """Simple rolling-window limiter for chat request and token budgets."""

    WINDOW_SECONDS = 60.0
    DAY_WINDOW_SECONDS = 86400.0

    def __init__(
        self,
        requests_per_minute: int,
        requests_per_day: int,
        tokens_per_minute: int,
        tokens_per_day: int,
        encoding_name: str,
    ) -> None:
        """Initialize the limiter.

        Args:
            requests_per_minute: Maximum requests allowed per rolling minute.
            requests_per_day: Maximum requests allowed per rolling day.
            tokens_per_minute: Maximum estimated tokens allowed per rolling minute.
            tokens_per_day: Maximum estimated tokens allowed per rolling day.
            encoding_name: Tokenizer encoding used for rough token estimation.
        """

        self._requests_per_minute = requests_per_minute
        self._requests_per_day = requests_per_day
        self._tokens_per_minute = tokens_per_minute
        self._tokens_per_day = tokens_per_day
        self._events: Deque[Tuple[float, int]] = deque()
        self._day_events: Deque[Tuple[float, int]] = deque()
        self._encoding = None
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding(encoding_name)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def acquire(self, messages: List[Dict[str, str]]) -> None:
        """Block until the rolling minute budget can accommodate the request.

        Args:
            messages: Chat payload whose tokens are being estimated.
        """

        estimated_tokens = max(1, self._estimate_tokens(messages))
        while True:
            now = time.monotonic()
            self._prune(now)

            request_count = len(self._events)
            token_total = sum(tokens for _, tokens in self._events)
            day_request_count = len(self._day_events)
            day_token_total = sum(tokens for _, tokens in self._day_events)

            if day_request_count >= self._requests_per_day:
                raise RuntimeError("Configured daily chat request budget has been exhausted.")
            if day_token_total + estimated_tokens > self._tokens_per_day:
                raise RuntimeError("Configured daily chat token budget has been exhausted.")

            if (
                request_count < self._requests_per_minute
                and token_total + estimated_tokens <= self._tokens_per_minute
            ):
                self._events.append((now, estimated_tokens))
                self._day_events.append((now, estimated_tokens))
                return

            sleep_seconds = self._seconds_until_capacity(now, estimated_tokens, token_total)
            LOGGER.info("Waiting %.2f seconds for the configured chat rate limit budget.", sleep_seconds)
            time.sleep(sleep_seconds)

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token usage for a chat payload.

        Args:
            messages: Chat message payload.

        Returns:
            Approximate token count.
        """

        combined_text = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
        )
        if self._encoding is None:
            return max(1, len(combined_text) // 4)
        return len(self._encoding.encode(combined_text))

    def _prune(self, now: float) -> None:
        """Discard events outside the rolling minute window.

        Args:
            now: Current monotonic timestamp.
        """

        while self._events and now - self._events[0][0] >= self.WINDOW_SECONDS:
            self._events.popleft()
        while self._day_events and now - self._day_events[0][0] >= self.DAY_WINDOW_SECONDS:
            self._day_events.popleft()

    def _seconds_until_capacity(self, now: float, estimated_tokens: int, token_total: int) -> float:
        """Compute the sleep time needed to satisfy the rate budget.

        Args:
            now: Current monotonic timestamp.
            estimated_tokens: Tokens required by the pending request.
            token_total: Tokens already consumed in the active window.

        Returns:
            Positive sleep duration in seconds.
        """

        wait_candidates: List[float] = []
        if self._events and len(self._events) >= self._requests_per_minute:
            wait_candidates.append(self.WINDOW_SECONDS - (now - self._events[0][0]) + 0.01)

        if self._events and token_total + estimated_tokens > self._tokens_per_minute:
            running_total = token_total + estimated_tokens
            for timestamp, tokens in self._events:
                running_total -= tokens
                if running_total <= self._tokens_per_minute:
                    wait_candidates.append(self.WINDOW_SECONDS - (now - timestamp) + 0.01)
                    break

        return max(0.05, max(wait_candidates, default=0.05))


class GroundedGenerator:
    """Generate strictly grounded answers from retrieved context."""

    def __init__(self, runtime_settings: Settings = settings, memory_window: int = MEMORY_WINDOW) -> None:
        """Initialize the generator.

        Args:
            runtime_settings: Application configuration.
            memory_window: Number of prior turns to retain.

        Raises:
            RuntimeError: If the OpenAI client dependency is unavailable.
        """

        if OpenAI is None:
            raise RuntimeError("The openai package is required for answer generation.")

        api_key = runtime_settings.LLM_API_KEY
        if not api_key and not runtime_settings.LLM_BASE_URL:
            raise RuntimeError("RAG_LLM_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY must be set when using the generator.")

        self._settings = runtime_settings
        self._memory_window = max(0, memory_window)
        self._conversation_history: List[Dict[str, str]] = []
        self._client = OpenAI(
            api_key=api_key or "local-placeholder-key",
            base_url=runtime_settings.LLM_BASE_URL,
            timeout=runtime_settings.REQUEST_TIMEOUT_SECONDS,
        )
        self._rate_limiter = _LLMRateLimiter(
            requests_per_minute=runtime_settings.LLM_MAX_REQUESTS_PER_MINUTE,
            requests_per_day=runtime_settings.LLM_MAX_REQUESTS_PER_DAY,
            tokens_per_minute=runtime_settings.LLM_MAX_TOKENS_PER_MINUTE,
            tokens_per_day=runtime_settings.LLM_MAX_TOKENS_PER_DAY,
            encoding_name=runtime_settings.TIKTOKEN_ENCODING,
        )

    def build_retrieval_query(self, question: str) -> str:
        """Augment retrieval with recent conversational context.

        Args:
            question: Current user question.

        Returns:
            Retrieval query containing recent turns and the latest question.
        """

        normalized_question = question.strip()
        if not normalized_question:
            return ""
        if self._memory_window == 0 or not self._conversation_history:
            return normalized_question

        recent_messages = self._conversation_history[-(self._memory_window * 2) :]
        history_lines = [
            f"{message['role'].capitalize()}: {message['content']}" for message in recent_messages
        ]
        history_lines.append(f"User: {normalized_question}")
        return "\n".join(history_lines)

    def stream_answer(self, question: str, context_block: str) -> Iterator[str]:
        """Stream a grounded answer for a question.

        Args:
            question: Current user question.
            context_block: Retrieved context block.

        Yields:
            Incremental text fragments from the model response.

        Raises:
            ValueError: If the question is empty.
        """

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")

        if "NO_RELEVANT_CONTEXT" in context_block:
            fallback = "I could not find the answer in the provided context."
            self._append_turn(normalized_question, fallback)
            yield fallback
            return

        messages = self._build_messages(normalized_question, context_block)
        self._rate_limiter.acquire(messages)
        stream = self._client.chat.completions.create(
            model=self._settings.LLM_MODEL,
            messages=messages,
            temperature=0.0,
            stream=True,
        )

        fragments: List[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text_fragment = delta.content or ""
            if not text_fragment:
                continue
            fragments.append(text_fragment)
            yield text_fragment

        final_answer = "".join(fragments).strip() or "I could not find the answer in the provided context."
        self._append_turn(normalized_question, final_answer)

    def clear_memory(self) -> None:
        """Clear the conversation history window."""

        self._conversation_history.clear()

    def _build_messages(self, question: str, context_block: str) -> List[Dict[str, str]]:
        """Construct chat messages for grounded generation.

        Args:
            question: Current user question.
            context_block: Retrieved context block.

        Returns:
            OpenAI chat message payload.
        """

        prompt_messages: List[Dict[str, str]] = [{"role": "system", "content": self._system_prompt()}]
        prompt_messages.extend(self._conversation_history[-(self._memory_window * 2) :])
        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    "Use the context block to answer the question.\n\n"
                    f"{context_block}\n\n"
                    f"Question: {question}"
                ),
            }
        )
        return prompt_messages

    def _append_turn(self, user_message: str, assistant_message: str) -> None:
        """Append one user/assistant turn and enforce the memory window.

        Args:
            user_message: User question text.
            assistant_message: Final assistant answer text.
        """

        if self._memory_window == 0:
            return

        self._conversation_history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self._conversation_history = self._conversation_history[-(self._memory_window * 2) :]

    def _system_prompt(self) -> str:
        """Return the grounding policy prompt.

        Returns:
            System prompt string.
        """

        return (
            "You are a retrieval-grounded assistant. Answer only with facts found in the provided context block. "
            "Do not use outside knowledge, do not infer beyond the context, and do not hallucinate. "
            "If the answer is missing, ambiguous, or only partially supported, explicitly say that the answer "
            "was not found in the provided context. Cite every factual statement using the exact bracketed source "
            "tags included in the context block, for example [Source: example.txt | Chunk: 3]."
        )