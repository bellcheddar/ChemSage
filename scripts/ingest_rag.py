#!/usr/bin/env python3
"""
ingest_rag.py — chunk, embed, and store the ChemSage corpus.

Phase 2 of PROJECT_PLAN.md. Ship this before any training: a working RAG layer on the
base model is already a usable product.

Usage:
    python scripts/ingest_rag.py --corpus data/corpus --store .chroma
    python scripts/ingest_rag.py --corpus data/corpus --store .chroma --reset
"""

import argparse
import hashlib
from pathlib import Path

import chromadb
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
EMBED_MODEL   = "BAAI/bge-base-en-v1.5"
UPSERT_BATCH  = 512


def load_documents(corpus_dir: Path):
    """Yield (source_path, text) for every document in the corpus.

    PDFs: full text via pypdf.
    CSV/TSV (SAR tables): one chunk per compound row, with the column header prepended
    so retrieval stays coherent regardless of which row is retrieved.
    TXT/MD: raw text.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}. Create it and add documents.")

    found = 0
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = _load_pdf(path)
            if text.strip():
                yield path, text
                found += 1
        elif suffix in (".csv", ".tsv"):
            for row_chunk in _load_sar_table(path):
                yield path, row_chunk
                found += 1
        elif suffix in (".txt", ".md"):
            text = path.read_text(errors="replace")
            if text.strip():
                yield path, text
                found += 1

    if found == 0:
        print(f"Warning: no documents found in {corpus_dir}. "
              "Drop PDFs, SAR CSVs, or text files there and re-run.")


def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"Warning: could not parse PDF {path.name}: {e}")
        return ""


def _load_sar_table(path: Path):
    """Yield one chunk per compound row: '[Table: name] Columns: … | Row: …'."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    except Exception as e:
        print(f"Warning: could not parse table {path.name}: {e}")
        return
    header = " | ".join(df.columns.tolist())
    for _, row in df.iterrows():
        row_text = " | ".join(row.tolist())
        yield f"[Table: {path.name}] Columns: {header}\nRow: {row_text}"


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Character-level sliding window chunker."""
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",      type=Path,   default=Path("data/corpus"))
    ap.add_argument("--store",       type=Path,   default=Path(".chroma"))
    ap.add_argument("--collection",  default="chemsage")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--reset",       action="store_true",
                    help="Delete and rebuild the collection from scratch")
    args = ap.parse_args()

    print(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    client = chromadb.PersistentClient(path=str(args.store))
    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted collection '{args.collection}'")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metas = [], [], []
    for source, text in load_documents(args.corpus):
        pieces = chunk(text)
        print(f"  {source.name}: {len(pieces)} chunks")
        for i, piece in enumerate(pieces):
            uid = hashlib.sha256(f"{source}:{i}:{piece[:80]}".encode()).hexdigest()[:16]
            ids.append(uid)
            texts.append(piece)
            metas.append({"source": str(source), "chunk": i})

    if not ids:
        print("No chunks to embed. Add documents to the corpus directory first.")
        return

    print(f"Embedding {len(ids)} chunks with {args.embed_model} ...")
    all_vecs = []
    for start in range(0, len(texts), 64):
        batch_vecs = embedder.encode(texts[start:start + 64], show_progress_bar=False).tolist()
        all_vecs.extend(batch_vecs)

    print(f"Upserting into Chroma collection '{args.collection}' ...")
    for start in range(0, len(ids), UPSERT_BATCH):
        collection.upsert(
            ids=ids[start:start + UPSERT_BATCH],
            embeddings=all_vecs[start:start + UPSERT_BATCH],
            documents=texts[start:start + UPSERT_BATCH],
            metadatas=metas[start:start + UPSERT_BATCH],
        )

    print(f"\nDone. {len(ids)} chunks in '{args.store}/{args.collection}'.")
    print("Exit test: query a corpus-only fact and confirm a grounded, sourced answer.")
    print("\nQuick query snippet:")
    print("  from sentence_transformers import SentenceTransformer")
    print("  import chromadb")
    print(f"  coll = chromadb.PersistentClient('{args.store}').get_collection('{args.collection}')")
    print("  emb  = SentenceTransformer('" + args.embed_model + "').encode(['your question']).tolist()")
    print("  print(coll.query(query_embeddings=emb, n_results=5))")


if __name__ == "__main__":
    main()
