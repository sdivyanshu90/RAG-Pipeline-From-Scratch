"""Token-aware document loading and recursive chunking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency availability is environmental.
    tiktoken = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of text prepared for indexing.

    Attributes:
        chunk_id: Stable UUID for the chunk payload.
        text: Chunk text content.
        metadata: Source metadata used for retrieval and citations.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, str | int]


class RecursiveTokenChunker:
    """Split documents into token-bounded overlapping chunks.

    The splitter preserves larger semantic boundaries first, then recursively
    falls back to smaller separators until each segment fits the configured
    token budget.
    """

    def __init__(
        self,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        encoding_name: str,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size_tokens: Maximum tokens per chunk.
            chunk_overlap_tokens: Token overlap retained between chunks.
            encoding_name: Tiktoken encoding used for token counting.
            separators: Optional recursive separator priority list.

        Raises:
            RuntimeError: If ``tiktoken`` is unavailable.
            ValueError: If the chunk configuration is invalid.
        """

        if tiktoken is None:
            raise RuntimeError(
                "tiktoken is required for token-aware chunking. Install dependencies from requirements.txt."
            )
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive.")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative.")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.separators = list(separators or ["\n\n", "\n", ". ", " ", ""])

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            LOGGER.warning("Unknown tiktoken encoding '%s'; falling back to cl100k_base.", encoding_name)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def load_directory(self, directory: Path, pattern: str) -> List[Tuple[Path, str]]:
        """Load matching UTF-8 text files from a directory.

        Args:
            directory: Directory containing source documents.
            pattern: Glob pattern for file discovery.

        Returns:
            Successfully decoded ``(path, text)`` tuples.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """

        if not directory.exists():
            raise FileNotFoundError(f"Document directory does not exist: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Document path is not a directory: {directory}")

        documents: List[Tuple[Path, str]] = []
        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue

            text = self._read_text_file(file_path)
            if text is None:
                continue
            if not text.strip():
                LOGGER.info("Skipping empty file %s.", file_path)
                continue
            documents.append((file_path, text))

        return documents

    def chunk_directory(self, directory: Path, pattern: str) -> List[DocumentChunk]:
        """Load and chunk all matching documents in a directory.

        Args:
            directory: Directory containing source documents.
            pattern: Glob pattern for file discovery.

        Returns:
            Flat list of generated chunk records.
        """

        chunks: List[DocumentChunk] = []
        for file_path, text in self.load_directory(directory, pattern):
            chunks.extend(self.chunk_document(text=text, source_name=file_path.name))
        return chunks

    def chunk_document(self, text: str, source_name: str) -> List[DocumentChunk]:
        """Split a single document into chunk records.

        Args:
            text: Full document text.
            source_name: Source filename stored in metadata.

        Returns:
            Chunk records with UUIDs and source metadata.
        """

        normalized_text = text.strip()
        if not normalized_text:
            return []

        atomic_segments = self._split_recursive(normalized_text, self.separators)
        merged_chunks = self._merge_segments(atomic_segments)
        return [
            DocumentChunk(
                chunk_id=str(uuid4()),
                text=chunk_text,
                metadata={"source": source_name, "chunk_index": chunk_index},
            )
            for chunk_index, chunk_text in enumerate(merged_chunks)
        ]

    def _read_text_file(self, file_path: Path) -> Optional[str]:
        """Read a single text file with graceful failure handling.

        Args:
            file_path: Source file path.

        Returns:
            Decoded text, or ``None`` if the file could not be read safely.
        """

        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                LOGGER.warning("Recovering non-UTF-8 bytes in %s with replacement characters.", file_path)
                return file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                LOGGER.warning("Skipping unreadable file %s: %s", file_path, exc)
                return None
        except OSError as exc:
            LOGGER.warning("Skipping unreadable file %s: %s", file_path, exc)
            return None

    def _split_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        """Recursively split text until each segment fits the token budget.

        Args:
            text: Input text to split.
            separators: Remaining separator priority list.

        Returns:
            Token-bounded text segments.
        """

        if self._token_count(text) <= self.chunk_size_tokens:
            return [text]
        if not separators:
            return self._split_by_tokens(text)

        separator = separators[0]
        if separator == "":
            return self._split_by_tokens(text)

        pieces = self._split_preserving_separator(text, separator)
        if len(pieces) <= 1:
            return self._split_recursive(text, separators[1:])

        flattened: List[str] = []
        for piece in pieces:
            stripped_piece = piece.strip()
            if not stripped_piece:
                continue
            if self._token_count(stripped_piece) <= self.chunk_size_tokens:
                flattened.append(stripped_piece)
            else:
                flattened.extend(self._split_recursive(stripped_piece, separators[1:]))
        return flattened

    def _split_preserving_separator(self, text: str, separator: str) -> List[str]:
        """Split text while retaining separators on intermediate segments.

        Args:
            text: Text to split.
            separator: Boundary string to preserve.

        Returns:
            Text pieces that retain separator context.
        """

        parts = text.split(separator)
        if len(parts) == 1:
            return [text]

        pieces: List[str] = []
        last_index = len(parts) - 1
        for index, part in enumerate(parts):
            if not part:
                continue
            suffix = separator if index < last_index else ""
            pieces.append(f"{part}{suffix}")
        return pieces or [text]

    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text directly on token boundaries.

        Args:
            text: Oversized text segment.

        Returns:
            Segments of at most ``chunk_size_tokens`` tokens.
        """

        token_ids = self.encoding.encode(text)
        segments: List[str] = []
        for start_index in range(0, len(token_ids), self.chunk_size_tokens):
            segment = self.encoding.decode(token_ids[start_index : start_index + self.chunk_size_tokens]).strip()
            if segment:
                segments.append(segment)
        return segments

    def _merge_segments(self, segments: Iterable[str]) -> List[str]:
        """Merge atomic segments into overlapping final chunks.

        Args:
            segments: Token-bounded atomic segments.

        Returns:
            Final chunk strings.
        """

        finalized_chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for segment in segments:
            segment_text = segment.strip()
            if not segment_text:
                continue

            segment_tokens = self._token_count(segment_text)
            if not current_parts:
                current_parts = [segment_text]
                current_tokens = segment_tokens
                continue

            if current_tokens + segment_tokens <= self.chunk_size_tokens:
                current_parts.append(segment_text)
                current_tokens += segment_tokens
                continue

            current_chunk = " ".join(current_parts).strip()
            if current_chunk:
                finalized_chunks.append(current_chunk)

            overlap_budget = max(0, self.chunk_size_tokens - segment_tokens)
            overlap_tokens = min(self.chunk_overlap_tokens, overlap_budget)
            overlap_text = self._tail_by_tokens(current_chunk, overlap_tokens)
            current_parts = [part for part in (overlap_text, segment_text) if part]
            current_tokens = self._token_count(" ".join(current_parts))

        trailing_chunk = " ".join(current_parts).strip()
        if trailing_chunk:
            finalized_chunks.append(trailing_chunk)
        return finalized_chunks

    def _tail_by_tokens(self, text: str, max_tokens: int) -> str:
        """Extract the trailing token window from text.

        Args:
            text: Input text span.
            max_tokens: Maximum number of tail tokens to keep.

        Returns:
            Decoded tail span.
        """

        if max_tokens <= 0:
            return ""
        token_ids = self.encoding.encode(text)
        if not token_ids:
            return ""
        return self.encoding.decode(token_ids[-max_tokens:]).strip()

    def _token_count(self, text: str) -> int:
        """Count tokens in a text span.

        Args:
            text: Input text.

        Returns:
            Number of tokens produced by the configured encoding.
        """

        return len(self.encoding.encode(text))