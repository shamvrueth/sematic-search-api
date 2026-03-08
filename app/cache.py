import time
import uuid
from typing import Optional
import numpy as np

MIN_CLUSTER_ENTRIES = 3

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {similarity_threshold}")

        self.threshold = similarity_threshold

        self.cache = {}
        self.cluster_index = {}

        self.hit_count = 0
        self.miss_count = 0
    
    def lookup(self, query_embedding: np.ndarray, dominant: int) -> Optional[dict]:
        if not self.cache:
            self.miss_count += 1
            return None

        ids = self.cluster_index.get(dominant, [])

        if len(ids) >= MIN_CLUSTER_ENTRIES:
            candidate_ids = ids
        else:
            candidate_ids = list(self.cache.keys())

        if not candidate_ids:
            self.miss_count += 1
            return None
        candidate_embeddings = np.stack([self.cache[eid]["embedding"] for eid in candidate_ids])

        sims = candidate_embeddings @ query_embedding # vectorised dot product
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            self.hit_count += 1
            matched_entry = self.cache[candidate_ids[best_idx]]
            return {**matched_entry, "similarity_score": best_sim}

        self.miss_count += 1
        return None
    
    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: str,
        dominant: int,
        probs: Optional[list] = None,
    ) -> str:
        
        entry_id = str(uuid.uuid4())
        entry = {
            "entry_id": entry_id,
            "query": query,
            "embedding": query_embedding,    # stored as numpy array
            "result": result,
            "dominant": dominant,
            "cluster_probs": probs or [],
            "timestamp": time.time(),
        }
        self.cache[entry_id] = entry
        if dominant not in self.cluster_index:
            self.cluster_index[dominant] = []
        self.cluster_index[dominant].append(entry_id)

        return entry_id
    
    def flush(self) -> None:  # clear all cache
        self.cache.clear()
        self.cluster_index.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    @property
    def stats(self) -> dict:
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_count / total, 4) if total > 0 else 0.0,
            "threshold": self.threshold,
            "clusters_populated": len(self.cluster_index),
        }
    
    @property
    def size(self) -> int:
        return len(self.cache)
    
    def simulate_threshold(
        self,
        queries_and_embeddings,
        threshold: float,
    ) -> dict:
        
        hits = 0
        misses = 0
        best_sims = []

        # build a temporary cache from first half, query with second half
        temp_cache = []
        for i, (query, emb, cluster) in enumerate(queries_and_embeddings):
            if i < len(queries_and_embeddings) // 2:
                temp_cache.append(emb)
                continue

            if not temp_cache:
                misses += 1
                continue

            cache_matrix = np.stack(temp_cache)
            sims = cache_matrix @ emb
            best_sim = float(np.max(sims))
            best_sims.append(best_sim)

            if best_sim >= threshold:
                hits += 1
            else:
                misses += 1

        total = hits + misses
        return {
            "threshold": threshold,
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            "n_hits": hits,
            "n_misses": misses,
            "avg_best_similarity": round(float(np.mean(best_sims)), 4) if best_sims else 0.0,
        }
    
def assign_cluster(
    query_embedding: np.ndarray,
    matrix: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
):
    sims = corpus_embeddings @ query_embedding  # shape (N,)
    top_k_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_k_sims = sims[top_k_idx]

    # softmax-weight the sims so they sum to 1
    weights = np.exp(top_k_sims - np.max(top_k_sims))
    weights = weights / weights.sum()

    # weighted average of cluster probability vectors
    neighbour_probs = matrix[top_k_idx]  # shape (top_k, K)
    query_probs = (weights[:, None] * neighbour_probs).sum(axis=0)  # shape (K,)

    dominant = int(np.argmax(query_probs))
    return dominant, query_probs.tolist()

if __name__ == "__main__":
    print("=== Semantic Cache Self-Test ===\n")
    cache = SemanticCache(similarity_threshold=0.85)

    # Simulate some embeddings (random unit vectors for testing)
    rng = np.random.default_rng(42)

    def random_unit_vector(dim=384):
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    # Store a query
    q1_emb = random_unit_vector()
    cache.store(
        query="What is the best rocket fuel?",
        query_embedding=q1_emb,
        result="Liquid hydrogen and liquid oxygen are common rocket propellants.",
        dominant=1,
    )
    # Test 1: near-identical query (slight noise) -> should HIT
    noise = random_unit_vector() * 0.05
    q2_emb = q1_emb + noise
    q2_emb = q2_emb / np.linalg.norm(q2_emb)
    result = cache.lookup(q2_emb, dominant=1)
    print(f"Test 1 (similar query): {'HIT' if result else 'MISS'}")
    if result:
        print(f"  similarity={result['similarity_score']:.4f}")
        print(f"  matched: '{result['query']}'")

    # Test 2: completely different query -> should MISS
    q3_emb = random_unit_vector()
    result = cache.lookup(q3_emb, dominant=5)
    print(f"\nTest 2 (different query): {'HIT' if result else 'MISS'}")

    print(f"\nCache stats: {cache.stats}")

    # Threshold exploration
    print("\n=== Threshold Exploration ===")
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    base = random_unit_vector()
    test_pairs = []
    for noise_level in [0.02, 0.05, 0.10, 0.20, 0.40]:
        noisy = base + random_unit_vector() * noise_level
        noisy = noisy / np.linalg.norm(noisy)
        sim = float(base @ noisy)
        print(f"  noise={noise_level:.2f}  cosine_sim={sim:.4f}  "
              + "  ".join(
                  f"t={t}:{'HIT' if sim >= t else 'miss'}"
                  for t in thresholds
              ))