from pathlib import Path
from typing import Dict, List, Optional
import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings
from tqdm import tqdm

COLLECTION_NAME = "newsgroups_corpus"
UPSERT_BATCH_SIZE = 500

class VectorStore:
    def __init__(self, persist_dir: str = "./embeddings/chromadb"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine distance for HNSW index
        )

    def ingest(self, df: pd.DataFrame, embeddings: np.ndarray, overwrite: bool = False) -> None:
        if overwrite:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        existing_count = self.collection.count()
        for start in tqdm(range(0, len(df), UPSERT_BATCH_SIZE), desc="Upserting batches"):
            end = min(start + UPSERT_BATCH_SIZE, len(df))
            batch_df = df.iloc[start:end]
            batch_embs = embeddings[start:end]

            ids = batch_df["doc_id"].tolist()
            documents = batch_df["text"].tolist()

            # Build metadata dicts — include all fields useful for filtered retrieval
            metadatas = []
            for _, row in batch_df.iterrows():
                meta: Dict = {
                    "category": row["category"],
                    "category_int": int(row["category_int"]),
                    "word_count": int(row["word_count"]),
                }
                # Cluster fields added by Part 2 — include if present
                if "dominant_cluster" in row and pd.notna(row.get("dominant_cluster")):
                    meta["dominant_cluster"] = int(row["dominant_cluster"])
                metadatas.append(meta)

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embs.tolist(),
                documents=documents,
                metadatas=metadatas,
            )

    def update_cluster_metadata(self, doc_ids: List[str], dominant_clusters: List[int],) -> None:
        for start in tqdm(range(0, len(doc_ids), UPSERT_BATCH_SIZE), desc="Updating"):
            end = min(start + UPSERT_BATCH_SIZE, len(doc_ids))
            batch_ids = doc_ids[start:end]

            # fetch existing metadata to merge, not overwrite
            existing = self.collection.get(ids=batch_ids, include=["metadatas"])
            updated_metadatas = []
            for meta, cluster in zip(existing["metadatas"], dominant_clusters[start:end]):
                meta["dominant_cluster"] = int(cluster)
                updated_metadatas.append(meta)

            self.collection.update(ids=batch_ids, metadatas=updated_metadatas)

    def query(self, query_embedding: np.ndarray, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        kwargs = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        # chromaDB cosine "distance" = 1 - cosine similarity
        results["similarities"] = [
            [round(1 - d, 4) for d in dist_list]
            for dist_list in results["distances"]
        ]

        return results
    
    def query_by_cluster(self, query_embedding: np.ndarray, cluster_id: int, n_results: int = 5) -> Dict:
        return self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where={"dominant_cluster": cluster_id},
        )
    
    def count(self) -> int:
        return self.collection.count()

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        result = self.collection.get(
            ids=[doc_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        return {
            "doc_id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
            "embedding": np.array(result["embeddings"][0], dtype=np.float32),
        }
    
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from embedder import load_embeddings

    # load corpus and embeddings
    corpus_path = Path("./data/processed/corpus.parquet")
    if not corpus_path.exists():
        print("Run data_prep.py first.")
        sys.exit(1)

    df = pd.read_parquet(corpus_path)
    embeddings, doc_ids = load_embeddings("./embeddings")

    assert list(df["doc_id"]) == doc_ids, "doc_id mismatch between corpus and embeddings"

    vs = VectorStore(persist_dir="./embeddings/chromadb")
    vs.ingest(df, embeddings)

    print("\nSanity check — querying: 'NASA space shuttle launch'")
    test_emb = embeddings[0]   # use first doc embedding as proxy
    results = vs.query(test_emb, n_results=3)
    for doc_id, sim, meta in zip(
        results["ids"][0], results["similarities"][0], results["metadatas"][0]
    ):
        print(f"  {doc_id}  sim={sim:.3f}  category={meta['category']}")

    print(f"\nVector store ready. Total documents: {vs.count()}")