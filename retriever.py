"""Query-time retrieval and strict context formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from config import SIMILARITY_THRESHOLD, TOP_K
from embedder import CachedEmbedder
from vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class RetrievalOutput:
    """Bundle search results with the formatted context block.

    Attributes:
        query: Query used for retrieval.
        results: Ranked vector search hits.
        context_block: Context block for grounded generation.
    """

    query: str
    results: List[SearchResult]
    context_block: str


class Retriever:
    """Retrieve relevant chunks and render them for grounded generation."""

    EMPTY_CONTEXT_BLOCK = "<CONTEXT>\nNO_RELEVANT_CONTEXT\n</CONTEXT>"

    def __init__(
        self,
        embedder: CachedEmbedder,
        vector_store: VectorStore,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        """Initialize the retriever.

        Args:
            embedder: Query embedding client.
            vector_store: Searchable vector index.
            top_k: Default maximum number of chunks to retrieve.
            similarity_threshold: Minimum cosine score accepted.
        """

        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query text.
            top_k: Optional query-specific top-k override.

        Returns:
            Ranked search results.
        """

        normalized_query = query.strip()
        if not normalized_query:
            return []

        query_vector = self._embedder.embed_query(normalized_query)
        return self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k or self._top_k,
            similarity_threshold=self._similarity_threshold,
        )

    def format_context(self, results: Sequence[SearchResult]) -> str:
        """Format results into a strict context block.

        Args:
            results: Retrieved chunk results.

        Returns:
            Context block ready for prompt injection.
        """

        if not results:
            return self.EMPTY_CONTEXT_BLOCK

        lines: List[str] = ["<CONTEXT>"]
        for result in results:
            source = result.metadata.get("source", "unknown")
            chunk_index = result.metadata.get("chunk_index", "unknown")
            lines.append(f"[Source: {source} | Chunk: {chunk_index}]")
            lines.append(result.text)
            lines.append("")
        lines.append("</CONTEXT>")
        return "\n".join(lines).strip()

    def retrieve_and_format(self, query: str, top_k: Optional[int] = None) -> RetrievalOutput:
        """Retrieve chunks and package them with their context block.

        Args:
            query: User query text.
            top_k: Optional query-specific top-k override.

        Returns:
            Retrieval output containing results and formatted context.
        """

        results = self.retrieve(query=query, top_k=top_k)
        return RetrievalOutput(query=query, results=results, context_block=self.format_context(results))