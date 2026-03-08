import argparse
import sys
from pathlib import Path
import pandas as pd

def run_ingestion(
    data_dir: str = None,
    overwrite: bool = False,
    min_words: int = 50,
    max_quote_ratio: float = 0.6,
):
    from prep import load_and_clean, save_processed
    from embedder import Embedder, embed_corpus, load_embeddings
    from vectorstore import VectorStore

    corpus_path = Path("./data/processed/corpus.parquet")
    if corpus_path.exists() and not overwrite:
        df = pd.read_parquet(corpus_path)
    
    else:
        df = load_and_clean(data_dir, min_words, max_quote_ratio)
        save_processed(df, "./data/processed")

    embedder = Embedder()
    embeddings = embed_corpus(
        df=df,
        embedder=embedder,
        out_dir="./embeddings",
        overwrite=overwrite,
    )

    doc_ids_from_file = None
    emb_path = Path("./embeddings/corpus_embeddings.npz")
    if emb_path.exists():
        import numpy as np
        data = np.load(emb_path, allow_pickle=True)
        doc_ids_from_file = data["doc_ids"].tolist()

    if doc_ids_from_file:
        assert list(df["doc_id"]) == doc_ids_from_file, (
            "FATAL: doc_id alignment mismatch between corpus and embeddings. "
            "Run with --overwrite to regenerate."
        )

    vs = VectorStore(persist_dir="./embeddings/chromadb")
    vs.ingest(df, embeddings, overwrite)
    return df, embeddings, vs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Part 1 ingestion pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"..\\data\\20_newsgroups",
        help=(
            "Path to local 20newsgroups root directory. "
            "Default assumes you run from the app\\ folder: ..\\data\\20_newsgroups"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-clean, re-embed, and re-ingest even if files exist.",
    )
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-quote-ratio", type=float, default=0.6)
    args = parser.parse_args()

    run_ingestion(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        min_words=args.min_words,
        max_quote_ratio=args.max_quote_ratio,
    )