"""CLI entrypoint for indexing documents and running RAG chat."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence

from chunker import RecursiveTokenChunker
from config import DEFAULT_DOCUMENT_GLOB, INDEX_DIR, settings
from embedder import CachedEmbedder
from generator import GroundedGenerator
from retriever import Retriever
from vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure application logging.

    Args:
        verbose: Whether to enable debug logging.
    """

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def build_index(documents_dir: Path, pattern: str, index_dir: Path) -> VectorStore:
    """Build and persist a vector index from local documents.

    Args:
        documents_dir: Directory containing source documents.
        pattern: Glob pattern for input discovery.
        index_dir: Directory to persist vector store artifacts.

    Returns:
        The populated vector store.

    Raises:
        RuntimeError: If no readable documents are found.
    """

    chunker = RecursiveTokenChunker(
        chunk_size_tokens=settings.CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        encoding_name=settings.TIKTOKEN_ENCODING,
    )
    chunks = chunker.chunk_directory(documents_dir, pattern)
    if not chunks:
        raise RuntimeError(
            f"No readable documents matched pattern {pattern!r} under directory {documents_dir}."
        )

    embedder = CachedEmbedder(settings)
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    store = VectorStore()
    store.add(embeddings, chunks)
    store.save(index_dir)
    LOGGER.info("Indexed %s chunk(s) into %s", len(chunks), index_dir)
    return store


def load_or_build_index(
    index_dir: Path,
    documents_dir: Optional[Path],
    pattern: str,
    rebuild: bool,
) -> VectorStore:
    """Load an existing index or build it from source documents.

    Args:
        index_dir: Directory containing persisted index artifacts.
        documents_dir: Optional source directory for building a new index.
        pattern: Glob pattern for input discovery.
        rebuild: Whether to force rebuilding the index.

    Returns:
        A ready-to-query vector store.

    Raises:
        FileNotFoundError: If no index exists and no documents directory is provided.
    """

    if rebuild:
        if documents_dir is None:
            raise FileNotFoundError("--docs-dir is required when --rebuild is used.")
        return build_index(documents_dir=documents_dir, pattern=pattern, index_dir=index_dir)

    try:
        store = VectorStore.load(index_dir)
        LOGGER.info("Loaded existing index from %s", index_dir)
        return store
    except FileNotFoundError:
        if documents_dir is None:
            raise FileNotFoundError(
                f"No existing index found in {index_dir}. Provide --docs-dir to build one."
            )
        return build_index(documents_dir=documents_dir, pattern=pattern, index_dir=index_dir)


def run_chat(
    index_dir: Path,
    documents_dir: Optional[Path],
    pattern: str,
    rebuild: bool,
    top_k: int,
    threshold: float,
    show_context: bool,
) -> int:
    """Run the interactive streaming chat loop.

    Args:
        index_dir: Directory containing persisted index artifacts.
        documents_dir: Optional source directory for building a missing index.
        pattern: Glob pattern for input discovery.
        rebuild: Whether to force index rebuilding before chat.
        top_k: Number of chunks to retrieve per question.
        threshold: Minimum cosine score accepted by retrieval.
        show_context: Whether to print the retrieved context block.

    Returns:
        Process exit code.
    """

    store = load_or_build_index(index_dir=index_dir, documents_dir=documents_dir, pattern=pattern, rebuild=rebuild)
    embedder = CachedEmbedder(settings)
    retriever = Retriever(embedder=embedder, vector_store=store, top_k=top_k, similarity_threshold=threshold)
    generator = GroundedGenerator(settings)

    print("Streaming RAG chat ready. Type 'exit' to quit or '/clear' to reset memory.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            print("Exiting.")
            return 0
        if question == "/clear":
            generator.clear_memory()
            print("Conversation memory cleared.")
            continue

        retrieval_query = generator.build_retrieval_query(question)
        retrieval_output = retriever.retrieve_and_format(query=retrieval_query, top_k=top_k)

        if retrieval_output.results:
            source_summary = ", ".join(
                f"{result.metadata.get('source', 'unknown')}#{result.metadata.get('chunk_index', 'unknown')}"
                for result in retrieval_output.results
            )
            LOGGER.info("Retrieved: %s", source_summary)
        else:
            LOGGER.info("Retrieved: no chunks above the similarity threshold.")

        if show_context:
            print("\nRetrieved Context:\n")
            print(retrieval_output.context_block)

        print("\nAssistant> ", end="", flush=True)
        for token in generator.stream_answer(question=question, context_block=retrieval_output.context_block):
            print(token, end="", flush=True)
        print()


def handle_index_command(args: argparse.Namespace) -> int:
    """Handle the ``index`` CLI command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    build_index(documents_dir=args.docs_dir, pattern=args.glob, index_dir=args.index_dir)
    return 0


def handle_chat_command(args: argparse.Namespace) -> int:
    """Handle the ``chat`` CLI command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """

    return run_chat(
        index_dir=args.index_dir,
        documents_dir=args.docs_dir,
        pattern=args.glob,
        rebuild=args.rebuild,
        top_k=args.top_k,
        threshold=args.threshold,
        show_context=args.show_context,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Production-grade RAG pipeline from scratch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a directory of text files.")
    index_parser.add_argument("--docs-dir", type=Path, required=True, help="Directory containing source documents.")
    index_parser.add_argument("--glob", default=DEFAULT_DOCUMENT_GLOB, help="Glob pattern for input files.")
    index_parser.add_argument("--index-dir", type=Path, default=INDEX_DIR, help="Output directory for the index.")
    index_parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    index_parser.set_defaults(handler=handle_index_command)

    chat_parser = subparsers.add_parser("chat", help="Start the streaming chat loop.")
    chat_parser.add_argument("--index-dir", type=Path, default=INDEX_DIR, help="Directory containing the index.")
    chat_parser.add_argument(
        "--docs-dir",
        type=Path,
        help="Optional source directory used to build the index when it does not exist.",
    )
    chat_parser.add_argument("--glob", default=DEFAULT_DOCUMENT_GLOB, help="Glob pattern for input files.")
    chat_parser.add_argument("--rebuild", action="store_true", help="Rebuild the index before chat.")
    chat_parser.add_argument("--top-k", type=int, default=settings.TOP_K, help="Retrieved chunks per query.")
    chat_parser.add_argument(
        "--threshold",
        type=float,
        default=settings.SIMILARITY_THRESHOLD,
        help="Minimum cosine score accepted for retrieval.",
    )
    chat_parser.add_argument("--show-context", action="store_true", help="Print the retrieved context block.")
    chat_parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    chat_parser.set_defaults(handler=handle_chat_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI application.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Process exit code.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=getattr(args, "verbose", False))

    try:
        return args.handler(args)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary.
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())