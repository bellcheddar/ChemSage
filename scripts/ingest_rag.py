#!/usr/bin/env python3
"""
ingest_rag.py — chunk, embed, and store the ChemSage corpus.

Phase 2 of PROJECT_PLAN.md. Build and ship this BEFORE any training: a working RAG layer on the
base model is already a usable product.

Usage:
    python scripts/ingest_rag.py --corpus data/corpus --store .chroma

Claude Code TODO:
  - Swap the placeholder embedder for either sentence-transformers (BAAI/bge-base-en-v1.5) or an
    Ollama embedding endpoint (nomic-embed-text).
  - Add CSV-aware chunking for SAR tables: keep each compound row intact and prepend the
    assay/target header to every chunk so retrieval stays coherent (see PROJECT_PLAN.md Phase 2).
  - Decide whether to own this pipeline or let AnythingLLM/Open WebUI manage the vector store.
"""

import argparse
from pathlib import Path

CHUNK_SIZE = 800        # characters; tune per corpus
CHUNK_OVERLAP = 120


def load_documents(corpus_dir: Path):
    """Yield (source_path, text) for each document. TODO: pypdf for PDFs, pandas for CSV/SAR."""
    raise NotImplementedError("Implement PDF + CSV loaders (pypdf, pandas).")


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Naive character chunker. TODO: replace with structure-aware chunking for tables."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def embed(chunks):
    """Return vectors for chunks. TODO: sentence-transformers or Ollama embeddings."""
    raise NotImplementedError("Wire up the embedder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--store", type=Path, default=Path(".chroma"))
    args = ap.parse_args()

    # TODO: open the vector store (chromadb.PersistentClient or lancedb.connect)
    for source, text in load_documents(args.corpus):
        for piece in chunk(text):
            vec = embed([piece])
            # TODO: upsert {id, vector, text, metadata={"source": str(source)}}
    print("Ingestion complete. Exit test: query a corpus-only fact and confirm a grounded answer.")


if __name__ == "__main__":
    main()
