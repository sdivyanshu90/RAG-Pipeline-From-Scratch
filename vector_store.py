"""Numpy-backed in-memory vector store with disk persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from chunker import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk paired with its cosine similarity score.

    Attributes:
        chunk_id: UUID associated with the chunk.
        text: Original chunk text.
        metadata: Stored chunk metadata.
        score: Cosine similarity score.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class VectorStore:
    """Store normalized embeddings and search them with vectorized cosine math."""

    MATRIX_FILENAME = "vectors.npy"
    RECORDS_FILENAME = "records.json"

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        records: Dict[int, Dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            matrix: Optional preloaded embedding matrix.
            records: Optional row-indexed payloads.

        Raises:
            ValueError: If the provided matrix is not 2-dimensional.
        """

        if matrix is None:
            self._matrix = np.empty((0, 0), dtype=np.float32)
        else:
            validated_matrix = np.asarray(matrix, dtype=np.float32)
            if validated_matrix.ndim != 2:
                raise ValueError("Vector matrix must be 2-dimensional.")
            self._matrix = self._normalize_rows(validated_matrix)

        self._records: Dict[int, Dict[str, Any]] = records or {}

    @property
    def size(self) -> int:
        """Return the number of stored vectors."""

        return int(self._matrix.shape[0])

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""

        if self._matrix.size == 0:
            return 0
        return int(self._matrix.shape[1])

    def add(self, embeddings: np.ndarray, chunks: Sequence[DocumentChunk]) -> None:
        """Append embeddings and their payloads to the store.

        Args:
            embeddings: Matrix of chunk embeddings.
            chunks: Chunk records aligned with ``embeddings``.

        Raises:
            ValueError: If the shapes or counts are inconsistent.
        """

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2-dimensional matrix.")
        if matrix.shape[0] != len(chunks):
            raise ValueError("The number of embeddings must match the number of chunks.")
        if matrix.shape[0] == 0:
            return

        normalized_matrix = self._normalize_rows(matrix)
        if self.size == 0:
            self._matrix = normalized_matrix
            start_index = 0
        else:
            if normalized_matrix.shape[1] != self.dimension:
                raise ValueError("Embedding dimension mismatch for vector store append.")
            start_index = self.size
            self._matrix = np.vstack((self._matrix, normalized_matrix)).astype(np.float32)

        for offset, chunk in enumerate(chunks):
            row_index = start_index + offset
            self._records[row_index] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }

    def search(self, query_vector: np.ndarray, top_k: int, similarity_threshold: float) -> List[SearchResult]:
        """Search the store using cosine similarity.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of hits to return.
            similarity_threshold: Minimum cosine score accepted.

        Returns:
            Ranked search results with scores.

        Raises:
            ValueError: If ``top_k`` is invalid or the query dimension is wrong.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.size == 0:
            return []

        normalized_query = self._normalize_query(query_vector)

        # Because both the matrix rows and the query are L2-normalized, the
        # matrix-vector dot product computes cosine similarity for every row.
        scores = self._matrix @ normalized_query

        candidate_count = min(top_k, scores.shape[0])
        if candidate_count == scores.shape[0]:
            candidate_indices = np.argsort(-scores)
        else:
            # argpartition isolates the top-k region in O(n) time before the
            # smaller candidate set is fully sorted by descending score.
            candidate_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]

        results: List[SearchResult] = []
        for row_index in candidate_indices.tolist():
            score = float(scores[row_index])
            if score < similarity_threshold:
                continue

            record = self._records[row_index]
            results.append(
                SearchResult(
                    chunk_id=str(record["chunk_id"]),
                    text=str(record["text"]),
                    metadata=dict(record["metadata"]),
                    score=score,
                )
            )
        return results

    def save(self, directory: Path) -> None:
        """Persist the vector matrix and metadata payloads to disk.

        Args:
            directory: Target directory for store artifacts.
        """

        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / self.MATRIX_FILENAME, self._matrix)

        payload = {
            "size": self.size,
            "dimension": self.dimension,
            "records": [
                {"index": row_index, **record} for row_index, record in sorted(self._records.items())
            ],
        }
        records_path = directory / self.RECORDS_FILENAME
        temporary_path = records_path.with_suffix(f"{records_path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_path.replace(records_path)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a persisted vector store from disk.

        Args:
            directory: Directory containing vector store artifacts.

        Returns:
            Loaded vector store.

        Raises:
            FileNotFoundError: If the expected artifacts are missing.
        """

        matrix_path = directory / cls.MATRIX_FILENAME
        records_path = directory / cls.RECORDS_FILENAME
        if not matrix_path.exists() or not records_path.exists():
            raise FileNotFoundError(f"Vector store artifacts not found in {directory}")

        matrix = np.load(matrix_path)
        if matrix.ndim == 1:
            matrix = np.atleast_2d(matrix)

        payload = json.loads(records_path.read_text(encoding="utf-8"))
        record_items = payload.get("records", []) if isinstance(payload, dict) else []

        records: Dict[int, Dict[str, Any]] = {}
        for item in record_items:
            if not isinstance(item, dict):
                continue
            row_index = int(item["index"])
            records[row_index] = {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
            }

        return cls(matrix=np.asarray(matrix, dtype=np.float32), records=records)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize each row in an embedding matrix.

        Args:
            matrix: Raw embedding matrix.

        Returns:
            Row-wise unit vectors.

        Raises:
            ValueError: If any row has zero norm.
        """

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Cannot store zero-length embedding vectors.")
        return (matrix / norms).astype(np.float32)

    def _normalize_query(self, query_vector: np.ndarray) -> np.ndarray:
        """Normalize a query vector before search.

        Args:
            query_vector: Raw query embedding.

        Returns:
            Unit-length query vector.

        Raises:
            ValueError: If the vector shape or dimension is invalid.
        """

        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self.dimension and vector.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, received {vector.shape[0]}."
            )
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("Cannot search with a zero-length query vector.")
        return (vector / norm).astype(np.float32)