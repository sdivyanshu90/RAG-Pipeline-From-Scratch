"""Embedding generation, retry logic, and local JSON caching."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from config import Settings, settings

try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - dependency availability is environmental.
    APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = None
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - dependency availability is environmental.
    SentenceTransformer = None


LOGGER = logging.getLogger(__name__)


class CachedEmbedder:
    """Generate normalized embeddings with persistent hash-based caching."""

    def __init__(self, runtime_settings: Settings = settings) -> None:
        """Initialize the embedder.

        Args:
            runtime_settings: Application configuration.
        """

        self._settings = runtime_settings
        self._cache_path = runtime_settings.EMBEDDING_CACHE_PATH
        self._cache: Dict[str, List[float]] = self._load_cache(self._cache_path)
        self._local_model: Optional[SentenceTransformer] = None
        self._openai_client = self._build_openai_client()

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            text: Query text.

        Returns:
            A unit-normalized query embedding.
        """

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple texts with cache reuse.

        Args:
            texts: Input texts to embed.

        Returns:
            Matrix of normalized embeddings aligned to the input order.

        Raises:
            ValueError: If any text is empty after normalization.
            RuntimeError: If the embedding provider returns an unexpected count.
        """

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        normalized_inputs = [self._sanitize_text(text) for text in texts]
        ordered_hashes: List[str] = []
        missing_by_hash: Dict[str, str] = {}

        for text in normalized_inputs:
            text_hash = self._hash_text(text)
            ordered_hashes.append(text_hash)
            if text_hash not in self._cache and text_hash not in missing_by_hash:
                missing_by_hash[text_hash] = text

        if missing_by_hash:
            fetched_vectors = self._embed_missing_texts(list(missing_by_hash.values()))
            if len(fetched_vectors) != len(missing_by_hash):
                raise RuntimeError("Embedding provider returned an unexpected number of vectors.")

            for text, vector in zip(missing_by_hash.values(), fetched_vectors, strict=True):
                self._cache[self._hash_text(text)] = self._normalize_vector(vector).astype(np.float32).tolist()
            self._save_cache(self._cache_path, self._cache)

        ordered_vectors = [np.asarray(self._cache[text_hash], dtype=np.float32) for text_hash in ordered_hashes]
        return np.vstack(ordered_vectors).astype(np.float32)

    def _build_openai_client(self) -> Optional[OpenAI]:
        """Create an OpenAI-compatible client when configured.

        Returns:
            Configured client, or ``None`` for local embedding mode.

        Raises:
            RuntimeError: If OpenAI mode is selected but dependencies or credentials are missing.
        """

        if self._settings.EMBEDDING_PROVIDER != "openai":
            return None
        if OpenAI is None:
            raise RuntimeError("The openai package is required for OpenAI embedding mode.")

        api_key = self._settings.EMBEDDING_API_KEY
        if not api_key and not self._settings.EMBEDDING_BASE_URL:
            raise RuntimeError(
                "RAG_EMBEDDING_API_KEY or OPENAI_API_KEY must be set when using OpenAI embedding mode."
            )

        return OpenAI(
            api_key=api_key or "local-placeholder-key",
            base_url=self._settings.EMBEDDING_BASE_URL,
            timeout=self._settings.REQUEST_TIMEOUT_SECONDS,
        )

    def _embed_missing_texts(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Embed cache misses in deterministic batches.

        Args:
            texts: Unique texts absent from the cache.

        Returns:
            Raw embedding vectors aligned to the input order.
        """

        vectors: List[np.ndarray] = []
        for batch in self._batched(texts, self._settings.EMBEDDING_BATCH_SIZE):
            if self._settings.EMBEDDING_PROVIDER == "openai":
                vectors.extend(self._with_retry(lambda: self._embed_batch_openai(batch)))
            elif self._settings.EMBEDDING_PROVIDER == "sentence-transformers":
                vectors.extend(self._embed_batch_local(batch))
            else:
                raise ValueError(f"Unsupported embedding provider: {self._settings.EMBEDDING_PROVIDER}")
        return vectors

    def _embed_batch_openai(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Call the OpenAI embedding API for one batch.

        Args:
            texts: Batch texts.

        Returns:
            Raw embedding vectors from the service.
        """

        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not configured.")

        response = self._openai_client.embeddings.create(model=self._settings.EMBEDDING_MODEL, input=list(texts))
        ordered_data = sorted(response.data, key=lambda item: item.index)
        return [np.asarray(item.embedding, dtype=np.float32) for item in ordered_data]

    def _embed_batch_local(self, texts: Sequence[str]) -> List[np.ndarray]:
        """Generate local embeddings with sentence-transformers.

        Args:
            texts: Batch texts.

        Returns:
            Raw embedding vectors from the local model.

        Raises:
            RuntimeError: If local embedding dependencies are unavailable.
        """

        model = self._get_local_model()
        embeddings = model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [np.asarray(vector, dtype=np.float32) for vector in embeddings]

    def _get_local_model(self) -> SentenceTransformer:
        """Lazily initialize the local embedding model.

        Returns:
            Initialized ``SentenceTransformer`` model.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required when RAG_EMBEDDING_PROVIDER=sentence-transformers."
            )
        if self._local_model is None:
            self._local_model = SentenceTransformer(self._settings.EMBEDDING_MODEL)
        return self._local_model

    def _with_retry(self, operation: Callable[[], List[np.ndarray]]) -> List[np.ndarray]:
        """Execute one embedding operation with exponential backoff.

        Args:
            operation: Zero-argument callable performing one API request.

        Returns:
            The successful operation result.

        Raises:
            RuntimeError: If all retry attempts fail.
        """

        delay_seconds = self._settings.RETRY_BASE_DELAY_SECONDS
        last_error: Optional[BaseException] = None

        for attempt in range(1, self._settings.RETRY_MAX_ATTEMPTS + 1):
            try:
                return operation()
            except Exception as exc:  # pragma: no cover - depends on external service behavior.
                if not self._is_retryable_openai_error(exc):
                    raise

                last_error = exc
                if attempt == self._settings.RETRY_MAX_ATTEMPTS:
                    break

                jitter = random.uniform(0.0, 0.25)
                sleep_seconds = delay_seconds + jitter
                LOGGER.warning(
                    "Embedding request failed on attempt %s/%s with %s. Retrying in %.2f seconds.",
                    attempt,
                    self._settings.RETRY_MAX_ATTEMPTS,
                    exc.__class__.__name__,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
                delay_seconds *= 2

        raise RuntimeError("Embedding request failed after all retry attempts.") from last_error

    def _is_retryable_openai_error(self, exc: BaseException) -> bool:
        """Return whether an exception should trigger an embedding retry.

        Args:
            exc: Raised exception.

        Returns:
            ``True`` when the exception is a known transient OpenAI failure.
        """

        retryable_types = tuple(
            error_type
            for error_type in (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
            if error_type is not None
        )
        return bool(retryable_types) and isinstance(exc, retryable_types)

    def _load_cache(self, cache_path: Path) -> Dict[str, List[float]]:
        """Load the persistent embedding cache from disk.

        Args:
            cache_path: Cache JSON file path.

        Returns:
            Mapping from text hash to embedding vector.
        """

        if not cache_path.exists():
            return {}

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring corrupted embedding cache %s: %s", cache_path, exc)
            return {}

        if not isinstance(payload, dict):
            LOGGER.warning("Ignoring embedding cache %s because it does not contain a JSON object.", cache_path)
            return {}

        cache: Dict[str, List[float]] = {}
        for text_hash, vector in payload.items():
            if isinstance(text_hash, str) and isinstance(vector, list):
                cache[text_hash] = [float(value) for value in vector]
        return cache

    def _save_cache(self, cache_path: Path, cache_payload: Dict[str, List[float]]) -> None:
        """Persist the embedding cache atomically.

        Args:
            cache_path: Cache JSON file path.
            cache_payload: Cache mapping to persist.
        """

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        temporary_path.replace(cache_path)

    def _sanitize_text(self, text: str) -> str:
        """Normalize whitespace and reject empty inputs.

        Args:
            text: Raw text.

        Returns:
            Stripped text.

        Raises:
            ValueError: If the normalized text is empty.
        """

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Cannot embed empty text.")
        return normalized_text

    def _hash_text(self, text: str) -> str:
        """Compute a stable cache key for a text span.

        Args:
            text: Input text.

        Returns:
            SHA-256 hex digest.
        """

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector.

        Args:
            vector: Raw embedding vector.

        Returns:
            Unit-length vector.

        Raises:
            ValueError: If the vector norm is zero.
        """

        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero-length embedding vector.")
        return vector / norm

    def _batched(self, items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
        """Yield deterministic batches from a sequence.

        Args:
            items: Input items.
            batch_size: Maximum batch size.

        Yields:
            Consecutive list batches.
        """

        for start_index in range(0, len(items), batch_size):
            yield list(items[start_index : start_index + batch_size])