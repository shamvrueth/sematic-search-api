import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = 64):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalise — cosine sim = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
    
    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text], show_progress=False)[0]
    
def embed_corpus(df: pd.DataFrame, embedder: Embedder, out_dir: str = "./embeddings", overwrite: bool = False):
    out_path = Path(out_dir) / "corpus_embeddings.npz"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        data = np.load(out_path, allow_pickle=True)
        saved_ids = data["doc_ids"].tolist()
        embeddings = data["embeddings"]

        current_ids = df["doc_id"].tolist()
        if saved_ids == current_ids:
            return embeddings
        
    texts = df["text"].tolist()
    embeddings = embedder.embed(texts, show_progress=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        doc_ids=np.array(df["doc_id"].tolist()),
    )
    return embeddings

def load_embeddings(out_dir: str = "./embeddings"):
    out_path = Path(out_dir) / "corpus_embeddings.npz"
    if not out_path.exists():
        raise FileNotFoundError(
            f"No embeddings found at {out_path}. Run embed_corpus() first."
        )
    data = np.load(out_path, allow_pickle=True)
    return data["embeddings"], data["doc_ids"].tolist()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    corpus_path = Path("./data/processed/corpus.parquet")
    if not corpus_path.exists():
        print("Run data_prep.py first to generate the cleaned corpus.")
        sys.exit(1)

    df = pd.read_parquet(corpus_path)
    print(f"Loaded corpus: {len(df)} documents")

    embedder = Embedder()
    embeddings = embed_corpus(df, embedder, out_dir="./embeddings")
    print(f"\nEmbeddings ready: shape={embeddings.shape}")


