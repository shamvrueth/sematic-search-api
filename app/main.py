import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cache import SemanticCache, assign_cluster
from embedder import Embedder
from vectorstore import VectorStore

_embedder = None          # embedder wrapper
_vector_store = None      # vectorStore wrapper
_cache = None

_corpus_embeddings = None   # shape (N, 384)
_cluster_probs = None        # shape (N, K)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _vector_store, _cache, _corpus_embeddings, _cluster_probs
    _embedder = Embedder()
    _vector_store = VectorStore(persist_dir="./embeddings/chromadb")
    doc_count = _vector_store.count()
    try:
        emb_data = np.load("./embeddings/corpus_embeddings.npz")
        _corpus_embeddings = emb_data["embeddings"].astype(np.float32)

        _cluster_probs = np.load("./models/cluster_probs.npy").astype(np.float32)

    except FileNotFoundError as e:
        raise RuntimeError(f"Required data files missing: {e}")
    
    _cache = SemanticCache(similarity_threshold=0.85)
    yield

app = FastAPI(
    title="Semantic Search API",
    description=(
        "Semantic search over the 20 Newsgroups corpus with fuzzy GMM clustering "
        "and an in-memory semantic cache. Cache uses cosine similarity on "
        "L2-normalised MiniLM embeddings with cluster-scoped lookup."
    ),
    version="1.0",
    lifespan=lifespan,
)

# pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")


class SearchResult(BaseModel):
    doc_id: str
    category: str
    score: float          # cosine similarity (higher = more similar)
    text_preview: str 


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None      # which cached query was reused
    similarity_score: Optional[float] = None  # how similar to matched query
    dominant: int
    results: list[SearchResult]
    latency_ms: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float
    clusters_populated: int


class CacheFlushResponse(BaseModel):
    message: str
    entries_cleared: int

def _check_ready():
    if _embedder is None or _vector_store is None or _cache is None:
        raise HTTPException(status_code=503, detail="Service not initialised yet.")


def _search_chromadb(query_embedding: np.ndarray, n_results: int = 5) -> list[SearchResult]:
    results = _vector_store.query(query_embedding, n_results=n_results)

    search_results = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    sims = results["similarities"][0]

    for doc, meta, sim in zip(docs, metas, sims):
        search_results.append(
            SearchResult(
                doc_id=str(meta.get("doc_id", results["ids"][0][len(search_results)])),
                category=meta.get("category", "unknown"),
                score=sim,
                text_preview=doc[:300],
            )
        )
    return search_results

def _results_to_str(results: list[SearchResult]) -> str:
    return "\n---\n".join(
        f"[{r.doc_id}] ({r.category}, score={r.score})\n{r.text_preview}"
        for r in results
    )


def _str_to_results(cached_str: str) -> list[SearchResult]:
    results = []
    for block in cached_str.split("\n---\n"):
        if not block.strip():
            continue
        try:
            header, preview = block.split("\n", 1)
            # header format: [doc_id] (category, score=0.9123)
            doc_id = header.split("]")[0].lstrip("[")
            rest = header.split("(")[1].rstrip(")")
            category, score_part = rest.split(", score=")
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    category=category,
                    score=float(score_part),
                    text_preview=preview.strip(),
                )
            )
        except Exception:
            continue
    return results

# endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    _check_ready()
    t0 = time.perf_counter()

    query_text = request.query.strip()
    try:
        query_embedding = _embedder.embed_single(query_text)  # shape (384,)

        dominant, cluster_probs = assign_cluster(
            query_embedding,
            _cluster_probs,
            _corpus_embeddings,
            top_k=10,
        )

        cached = _cache.lookup(query_embedding, dominant)

        if cached:
            # cache HIT
            results = _str_to_results(cached["result"])
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return QueryResponse(
                query=query_text,
                cache_hit=True,
                matched_query=cached["query"],
                similarity_score=round(cached["similarity_score"], 4),
                dominant=dominant,
                results=results,
                latency_ms=latency,
            )

        # cache MISS
        results = _search_chromadb(query_embedding, n_results=5)
        result_str = _results_to_str(results)

        _cache.store(
            query=query_text,
            query_embedding=query_embedding,
            result=result_str,
            dominant=dominant,
            probs=cluster_probs,
        )

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            dominant=dominant,
            results=results,
            latency_ms=latency,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    _check_ready()
    s = _cache.stats
    return CacheStatsResponse(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
        threshold=s["threshold"],
        clusters_populated=s["clusters_populated"],
    )

@app.delete("/cache", response_model=CacheFlushResponse)
async def flush_cache():
    _check_ready()
    entries_before = _cache.size
    _cache.flush()
    return CacheFlushResponse(
        message="Cache flushed successfully.",
        entries_cleared=entries_before,
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chromadb_docs": _vector_store.count() if _vector_store else 0,
        "cache_entries": _cache.size if _cache else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
