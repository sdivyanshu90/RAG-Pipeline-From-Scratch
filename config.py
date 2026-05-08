"""Central configuration for the RAG pipeline.

This module keeps runtime settings in one place and supports environment
variable overrides for deployment without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _load_dotenv(dotenv_path: Path) -> None:
    """Load a minimal .env file into the process environment.

    Existing environment variables win over .env entries.

    Args:
        dotenv_path: Path to the .env file.
    """

    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Failed to read environment file: {dotenv_path}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip()
        if not env_key:
            continue

        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {'"', "'"}:
            env_value = env_value[1:-1]

        os.environ.setdefault(env_key, env_value)


def _read_int(name: str, default: int, minimum: int | None = None) -> int:
    """Read an integer environment variable with validation.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.
        minimum: Optional lower bound for accepted values.

    Returns:
        A validated integer.

    Raises:
        ValueError: If the provided value is not a valid integer or violates
            the configured lower bound.
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, received {raw_value!r}.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, received {value}.")

    return value


def _read_float(name: str, default: float, minimum: float | None = None) -> float:
    """Read a float environment variable with validation.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.
        minimum: Optional lower bound for accepted values.

    Returns:
        A validated float.

    Raises:
        ValueError: If the provided value is not a valid float or violates the
            configured lower bound.
    """

    raw_value = os.getenv(name)
    if raw_value is None:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, received {raw_value!r}.") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, received {value}.")

    return value


@dataclass(frozen=True)
class Settings:
    """Application settings for indexing and chat generation.

    Attributes:
        EMBEDDING_PROVIDER: Embedding backend. Supported values are
            ``openai`` and ``sentence-transformers``.
        EMBEDDING_MODEL: Name of the embedding model.
        LLM_MODEL: Name of the chat completion model.
        TIKTOKEN_ENCODING: Tokenizer encoding used for chunk sizing.
        CHUNK_SIZE_TOKENS: Maximum number of tokens per chunk.
        CHUNK_OVERLAP_TOKENS: Overlap retained between adjacent chunks.
        EMBEDDING_BATCH_SIZE: Number of texts to embed per request.
        TOP_K: Maximum number of chunks retrieved per query.
        SIMILARITY_THRESHOLD: Minimum cosine similarity required for a hit.
        MEMORY_WINDOW: Number of prior conversation turns to retain.
        RETRY_MAX_ATTEMPTS: Maximum API retry attempts on transient failures.
        RETRY_BASE_DELAY_SECONDS: Base delay used for exponential backoff.
        REQUEST_TIMEOUT_SECONDS: Network timeout for model calls.
        OPENAI_API_KEY: API key for OpenAI-compatible endpoints.
        OPENAI_BASE_URL: Optional custom base URL for OpenAI-compatible APIs.
        LLM_API_KEY: API key used by the chat generation backend.
        LLM_BASE_URL: Base URL used by the chat generation backend.
        EMBEDDING_API_KEY: API key used by the embedding backend.
        EMBEDDING_BASE_URL: Base URL used by the embedding backend.
        LLM_MAX_REQUESTS_PER_MINUTE: Request budget enforced for chat calls.
        LLM_MAX_REQUESTS_PER_DAY: Daily request budget enforced for chat calls.
        LLM_MAX_TOKENS_PER_MINUTE: Approximate token budget enforced for chat calls.
        LLM_MAX_TOKENS_PER_DAY: Daily token budget enforced for chat calls.
        INDEX_DIR: Directory for persisted vector index artifacts.
        EMBEDDING_CACHE_PATH: JSON cache for text-hash to embedding mappings.
        DEFAULT_DOCUMENT_GLOB: File glob used during indexing.
    """

    EMBEDDING_PROVIDER: str
    EMBEDDING_MODEL: str
    LLM_MODEL: str
    TIKTOKEN_ENCODING: str
    CHUNK_SIZE_TOKENS: int
    CHUNK_OVERLAP_TOKENS: int
    EMBEDDING_BATCH_SIZE: int
    TOP_K: int
    SIMILARITY_THRESHOLD: float
    MEMORY_WINDOW: int
    RETRY_MAX_ATTEMPTS: int
    RETRY_BASE_DELAY_SECONDS: float
    REQUEST_TIMEOUT_SECONDS: float
    OPENAI_API_KEY: str | None
    OPENAI_BASE_URL: str | None
    LLM_API_KEY: str | None
    LLM_BASE_URL: str | None
    EMBEDDING_API_KEY: str | None
    EMBEDDING_BASE_URL: str | None
    LLM_MAX_REQUESTS_PER_MINUTE: int
    LLM_MAX_REQUESTS_PER_DAY: int
    LLM_MAX_TOKENS_PER_MINUTE: int
    LLM_MAX_TOKENS_PER_DAY: int
    INDEX_DIR: Path
    EMBEDDING_CACHE_PATH: Path
    DEFAULT_DOCUMENT_GLOB: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            A fully validated settings instance.
        """

        project_root = Path(__file__).resolve().parent
        index_dir = Path(os.getenv("RAG_INDEX_DIR", project_root / "artifacts" / "index")).expanduser()
        embedding_cache_path = Path(
            os.getenv("RAG_EMBEDDING_CACHE_PATH", project_root / "artifacts" / "embedding_cache.json")
        ).expanduser()

        settings = cls(
            EMBEDDING_PROVIDER=os.getenv("RAG_EMBEDDING_PROVIDER", "sentence-transformers").strip().lower(),
            EMBEDDING_MODEL=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip(),
            LLM_MODEL=os.getenv("RAG_LLM_MODEL", "openai/gpt-oss-120b").strip(),
            TIKTOKEN_ENCODING=os.getenv("RAG_TIKTOKEN_ENCODING", "cl100k_base").strip(),
            CHUNK_SIZE_TOKENS=_read_int("RAG_CHUNK_SIZE_TOKENS", 400, minimum=32),
            CHUNK_OVERLAP_TOKENS=_read_int("RAG_CHUNK_OVERLAP_TOKENS", 60, minimum=0),
            EMBEDDING_BATCH_SIZE=_read_int("RAG_EMBEDDING_BATCH_SIZE", 32, minimum=1),
            TOP_K=_read_int("RAG_TOP_K", 5, minimum=1),
            SIMILARITY_THRESHOLD=_read_float("RAG_SIMILARITY_THRESHOLD", 0.2, minimum=0.0),
            MEMORY_WINDOW=_read_int("RAG_MEMORY_WINDOW", 4, minimum=0),
            RETRY_MAX_ATTEMPTS=_read_int("RAG_RETRY_MAX_ATTEMPTS", 5, minimum=1),
            RETRY_BASE_DELAY_SECONDS=_read_float("RAG_RETRY_BASE_DELAY_SECONDS", 1.0, minimum=0.0),
            REQUEST_TIMEOUT_SECONDS=_read_float("RAG_REQUEST_TIMEOUT_SECONDS", 60.0, minimum=1.0),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL"),
            LLM_API_KEY=os.getenv("RAG_LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"),
            LLM_BASE_URL=(
                os.getenv("RAG_LLM_BASE_URL")
                or os.getenv("GROQ_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.groq.com/openai/v1"
            ),
            EMBEDDING_API_KEY=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            EMBEDDING_BASE_URL=os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            LLM_MAX_REQUESTS_PER_MINUTE=_read_int("RAG_LLM_MAX_REQUESTS_PER_MINUTE", 30, minimum=1),
            LLM_MAX_REQUESTS_PER_DAY=_read_int("RAG_LLM_MAX_REQUESTS_PER_DAY", 1000, minimum=1),
            LLM_MAX_TOKENS_PER_MINUTE=_read_int("RAG_LLM_MAX_TOKENS_PER_MINUTE", 8000, minimum=1),
            LLM_MAX_TOKENS_PER_DAY=_read_int("RAG_LLM_MAX_TOKENS_PER_DAY", 200000, minimum=1),
            INDEX_DIR=index_dir,
            EMBEDDING_CACHE_PATH=embedding_cache_path,
            DEFAULT_DOCUMENT_GLOB=os.getenv("RAG_DOCUMENT_GLOB", "**/*.txt").strip(),
        )

        if settings.CHUNK_OVERLAP_TOKENS >= settings.CHUNK_SIZE_TOKENS:
            raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be smaller than RAG_CHUNK_SIZE_TOKENS.")

        if settings.EMBEDDING_PROVIDER not in {"openai", "sentence-transformers"}:
            raise ValueError(
                "RAG_EMBEDDING_PROVIDER must be either 'openai' or 'sentence-transformers'."
            )

        return settings


_load_dotenv(Path(__file__).resolve().parent / ".env")


settings = Settings.from_env()

EMBEDDING_PROVIDER = settings.EMBEDDING_PROVIDER
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
LLM_MODEL = settings.LLM_MODEL
TIKTOKEN_ENCODING = settings.TIKTOKEN_ENCODING
CHUNK_SIZE_TOKENS = settings.CHUNK_SIZE_TOKENS
CHUNK_OVERLAP_TOKENS = settings.CHUNK_OVERLAP_TOKENS
EMBEDDING_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE
TOP_K = settings.TOP_K
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD
MEMORY_WINDOW = settings.MEMORY_WINDOW
RETRY_MAX_ATTEMPTS = settings.RETRY_MAX_ATTEMPTS
RETRY_BASE_DELAY_SECONDS = settings.RETRY_BASE_DELAY_SECONDS
REQUEST_TIMEOUT_SECONDS = settings.REQUEST_TIMEOUT_SECONDS
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_BASE_URL = settings.OPENAI_BASE_URL
LLM_API_KEY = settings.LLM_API_KEY
LLM_BASE_URL = settings.LLM_BASE_URL
EMBEDDING_API_KEY = settings.EMBEDDING_API_KEY
EMBEDDING_BASE_URL = settings.EMBEDDING_BASE_URL
LLM_MAX_REQUESTS_PER_MINUTE = settings.LLM_MAX_REQUESTS_PER_MINUTE
LLM_MAX_REQUESTS_PER_DAY = settings.LLM_MAX_REQUESTS_PER_DAY
LLM_MAX_TOKENS_PER_MINUTE = settings.LLM_MAX_TOKENS_PER_MINUTE
LLM_MAX_TOKENS_PER_DAY = settings.LLM_MAX_TOKENS_PER_DAY
INDEX_DIR = settings.INDEX_DIR
EMBEDDING_CACHE_PATH = settings.EMBEDDING_CACHE_PATH
DEFAULT_DOCUMENT_GLOB = settings.DEFAULT_DOCUMENT_GLOB