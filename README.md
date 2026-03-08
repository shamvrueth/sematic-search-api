# Semantic Search over 20 Newsgroups

An end-to-end semantic search system built on the 20 Newsgroups dataset. Documents are embedded with a transformer model, indexed in a vector database, and soft-clustered with a Gaussian Mixture Model. A semantic cache reduces redundant computation by recognising paraphrased queries.

---

## Architecture

```
Raw text files
      │
      ▼
  prep.py          — Clean, filter, deduplicate → corpus.parquet (15,807 docs)
      │
      ▼
  embedder.py      — MiniLM-L6-v2 embeddings (384-dim, L2-normalised)
      │
      ▼
  vectorstore.py  — ChromaDB persistent collection (cosine HNSW index)
      │
      ▼
  cluster.py       — UMAP (384D→5D) + GMM (K=25, BIC-selected) → soft cluster assignments
      │
      ▼
  cache.py         — In-memory semantic cache (cluster-scoped cosine similarity)
      │
      ▼
  main.py          — FastAPI service (POST /query, GET /cache/stats, DELETE /cache)
```

---

## Part 1 — Data Preparation

### Cleaning pipeline (`prep.py`)

Raw files go through four filters in order:

**Header stripping** removes `From:`, `Subject:`, `Date:` and similar metadata lines. These are unambiguously non-content and would pollute embeddings with email infrastructure noise.

**Minimum word count (≥ 50 words)** removes stubs, signature-only posts, and one-liners. The threshold is a heuristic and is exposed as a parameter.

**Quote ratio filter (≤ 0.6)** discards posts where more than 60% of lines begin with `>` (quoted previous messages). Near-pure quote chains add no original content. This is the most debatable filter — some quoted discussions are meaningful — but the threshold is conservative and tunable.

**Exact deduplication (MD5)** removes bit-identical documents. These are cross-posted articles appearing in multiple newsgroups.

Result: 15,807 documents retained from 19,997 raw files.

### Embedding model

`all-MiniLM-L6-v2` — 384-dimensional output, fast on CPU (~20 minutes for 15k documents), trained on over 1 billion sentence pairs. The 384 dimensions are a fixed property of the model architecture. Embeddings are L2-normalised at generation time so cosine similarity reduces to a dot product at query time.

### Vector store

ChromaDB with a cosine HNSW index. Self-contained, no server required, supports metadata filtering, and persists to disk across restarts.

---

## Part 2 — Fuzzy Clustering

### Why fuzzy clustering

The 20 newsgroup labels are editorially defined. Real semantic structure is messier:

- `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, and `comp.os.ms-windows.misc` are semantically one hardware topic split across three labels.
- `talk.politics.misc` contains gun control, foreign policy, and social issues — semantically distinct topics under one label.
- A post about gun legislation genuinely belongs to both `talk.politics.guns` and `talk.politics.misc` simultaneously.

Hard clustering forces a binary assignment. GMM gives each document a full probability distribution over all clusters — a richer and more honest representation.

### Pipeline

**Dimensionality reduction (UMAP)** — Clustering directly on 384-dimensional embeddings fails due to the curse of dimensionality: in very high dimensions every point is roughly equidistant from every other. UMAP reduces to 5 dimensions while preserving local neighbourhood structure. A separate 2D projection is computed for visualisation only.

**K selection (BIC)** — K = 5, 10, 15, 20, 25, 30 are evaluated using BIC (Bayesian Information Criterion):

```
BIC = −2 × log_likelihood + n_parameters × log(n_documents)
```

BIC rewards fit quality but penalises model complexity. The BIC curve continues decreasing across the full tested range, indicating the corpus has richer semantic structure than K=30 captures. K=25 was selected as a practical compromise balancing interpretability against computation cost.

**GMM fitting** — `covariance_type="full"` allows each cluster blob to be any elliptical shape, not just axis-aligned. The EM algorithm alternates between the E step (soft cluster assignment probabilities) and M step (update cluster shapes using soft assignments) until convergence.

### Cluster results (K=25)

| Cluster | Dominant categories | Interpretation |
|---------|-------------------|----------------|
| 3 | rec.sport.hockey 776 | Hockey — very pure |
| 7 | talk.politics.mideast 207 | Middle East politics — very pure |
| 8 | sci.med 642 | Medicine — very pure |
| 9 | sci.crypt 619 | Cryptography — very pure |
| 15 | sci.space 610 | Space / NASA |
| 19 | soc.religion + alt.atheism mixed | Religion & belief — genuine boundary |
| 5 | politics + guns + crypt mixed | Civil liberties — genuine boundary |

The model recovered meaningful topic structure without ever seeing the labels. High-entropy boundary clusters (19, 5) represent genuinely cross-topic content — posts about encryption policy sit between cryptography and politics, gun legislation posts sit between firearms and politics.

### Entropy distribution

Shannon entropy measures assignment uncertainty per document. The distribution shows ~90% of documents with low entropy (clear assignment) and a meaningful tail of uncertain boundary documents. Maximum possible entropy for K=25 is ln(25) ≈ 3.22. The most uncertain documents score around 1.4, confirming the model is confident overall but appropriately uncertain at genuine topic boundaries.

---

## Part 3 — Semantic Cache

### Design

The cache stores query embeddings and returns a hit when a new query is semantically similar to a stored one — even if phrased differently. All state is in-memory. No Redis, no external libraries — pure Python and numpy.

**Data structure:**
```
primary store:   {entry_id: {query, embedding, result, dominant_cluster, timestamp}}
cluster index:   {cluster_id: [entry_id, entry_id, ...]}
```

### Cluster-scoped lookup

Naive lookup compares a new query against every cached entry — O(N). With cluster routing we compare only against entries in the same cluster — O(N/K). For K=25 this is a ~96% reduction at large cache sizes. This is also semantically justified: a query about space is extremely unlikely to match a cached query about hockey.

A cold-start fallback to full-cache search fires when a cluster has fewer than 3 cached entries.

### Similarity threshold

The threshold is the central tunable parameter:

| Threshold | Behaviour |
|-----------|-----------|
| 0.70 | Very permissive — cache acts as a coarse topic router |
| 0.80 | Near-paraphrases hit; loosely related queries miss |
| 0.85 | Our default — confirmed empirically on this corpus |
| 0.95 | Very strict — only near-identical rephrasings hit |

Empirical validation on this corpus: "NASA space shuttle launch mission" vs "rocket launch into orbit NASA" scores 0.708 — a miss at 0.85, correctly. "NASA shuttle launch mission" scores 0.991 — a hit.

**Latency impact:** cache miss ~1700ms (embedding + ChromaDB search), cache hit ~22-40ms. 40-80× speedup on hits.

### Query cluster assignment

At inference time re-running the GMM is too slow. Instead we use weighted nearest-neighbour interpolation: find the top-10 most similar corpus documents, softmax-weight their similarities, and take the weighted average of their GMM probability vectors. This gives a semantically grounded cluster assignment in ~2ms.

---

## Part 4 — FastAPI Service

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Semantic search with cache |
| `GET` | `/cache/stats` | Hit/miss statistics |
| `DELETE` | `/cache` | Flush cache and reset stats |
| `GET` | `/health` | Liveness check |

### POST /query

**Request:**
```json
{"query": "NASA space shuttle launch mission"}
```

**Cache miss response:**
```json
{
  "query": "NASA space shuttle launch mission",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "dominant_cluster": 15,
  "results": [{"doc_id": "doc_14011", "category": "sci.space", "score": 0.5421, "text_preview": "..."}],
  "latency_ms": 1713.7
}
```

**Cache hit response:**
```json
{
  "query": "NASA shuttle launch mission",
  "cache_hit": true,
  "matched_query": "NASA space shuttle launch mission",
  "similarity_score": 0.9906,
  "dominant_cluster": 15,
  "results": [...],
  "latency_ms": 21.93
}
```

---

## Running Locally

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run full ingestion pipeline
cd app
python ingest.py --data-dir ../data/20_newsgroups

# Run clustering
python cluster.py --k 25

# Start the API
uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

---

## Running with Docker

```bash
docker compose up --build
```

The container mounts `./embeddings`, `./models`, and `./data` from the host at runtime. Data is never baked into the image. Docker polls `GET /health` every 30 seconds with a 60-second grace period for model loading.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | MiniLM-L6-v2 | Fast CPU inference, strong semantic quality, 384-dim |
| Vector store | ChromaDB | Embedded, no server, cosine HNSW, persistent |
| Dim reduction | UMAP 5D | Avoids curse of dimensionality; preserves neighbourhood structure |
| Clustering | GMM full covariance | Flexible blob shapes; probabilistic output; statistically principled |
| K selection | BIC curve | Penalises complexity; evidence-based |
| Cache similarity | Cosine dot product | Single numpy call on L2-normalised vectors |
| Cache scope | Cluster-scoped + cold-start fallback | O(N/K) lookup; semantically justified |
| Cache threshold | 0.85 configurable | Near-paraphrases hit; loosely related queries miss |