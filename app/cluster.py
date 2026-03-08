import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import joblib
matplotlib.use("Agg")
from scipy.stats import entropy as scipy_entropy

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_components_2d: int = 2,
    random_state: int = 42,
    models_dir: str = "./models",
):
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    nd_path = Path(models_dir) / f"umap_{n_components}d.npy"
    twod_path = Path(models_dir) / "umap_2d.npy"
    if nd_path.exists():
        reduced_nd = np.load(nd_path)
    else:
        reducer_nd = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,      # local neighbourhood size — 15 is standard
            min_dist=0.1,        # how tightly to pack points
            metric="cosine",     # matches our embedding similarity metric
            random_state=random_state,
            verbose=True,
        )
        reduced_nd = reducer_nd.fit_transform(embeddings)
        np.save(nd_path, reduced_nd)

    if twod_path.exists():
        reduced_2d = np.load(twod_path)
    else:
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
            verbose=False,
        )
        reduced_2d = reducer_2d.fit_transform(embeddings)
        np.save(twod_path, reduced_2d)

    return reduced_nd, 

def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_components_2d: int = 2,
    random_state: int = 42,
    models_dir: str = "./models",
):
    nd_path = Path(models_dir) / f"umap_{n_components}d.npy"
    twod_path = Path(models_dir) / "umap_2d.npy"
    if nd_path.exists():
        reduced_nd = np.load(nd_path)
    
    else:
        reducer_nd = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,      # local neighbourhood size — 15 is standard
            min_dist=0.1,        # how tightly to pack points
            metric="cosine",     # matches our embedding similarity metric
            random_state=random_state,
            verbose=True,
        )
        reduced_nd = reducer_nd.fit_transform(embeddings)
        np.save(nd_path, reduced_nd)

    if twod_path.exists():
        reduced_2d = np.load(twod_path)
    else:
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=random_state,
            verbose=False,
        )
        reduced_2d = reducer_2d.fit_transform(embeddings)
        np.save(twod_path, reduced_2d)

    return reduced_nd, reduced_2d

def select_k_by_bic(
    reduced: np.ndarray,
    k_range: list = [5, 10, 15, 20, 25, 30],
    models_dir: str = "./models",
    random_state: int = 42,
):
    bic_path = Path(models_dir) / "bic_scores.json"

    if bic_path.exists():
        with open(bic_path) as f:
            bic_scores = {int(k): v for k, v in json.load(f).items()}
    else:
        bic_scores = {}
        for k in tqdm(k_range, desc="BIC evaluation"):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=random_state,
                max_iter=200,
                n_init=3,        # 3 random initialisations — takes best
            )
            gmm.fit(reduced)
            bic_scores[k] = float(gmm.bic(reduced))
    
        with open(bic_path, "w") as f:
                json.dump(bic_scores, f, indent=2)

    plot_bic(bic_scores, models_dir)
    ks = sorted(bic_scores.keys())
    bics = [bic_scores[k] for k in ks]
    drops = [bics[i] - bics[i+1] for i in range(len(bics)-1)]
    elbow_idx = drops.index(max(drops))
    best_k = ks[elbow_idx + 1]

    return best_k, bic_scores

def plot_bic(bic_scores: dict, models_dir: str) -> None:
    ks = sorted(bic_scores.keys())
    bics = [bic_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, bics, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("BIC score (lower = better)")
    ax.set_title("GMM BIC score vs K — elbow indicates best K")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(models_dir) / "bic_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

def fit_gmm(reduced: np.ndarray, k: int, models_dir: str = "./models", random_state: int = 42):
    model_path = Path(models_dir) / f"gmm_k{k}.joblib"

    if model_path.exists():
        gmm = joblib.load(model_path)
    else:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
            max_iter=200,
            n_init=5,
            verbose=1,
        )
        gmm.fit(reduced)
        joblib.dump(gmm, model_path)

    probs = gmm.predict_proba(reduced)

    return gmm, probs

def analyse_clusters(
    df: pd.DataFrame,
    probs: np.ndarray,
    reduced_2d: np.ndarray,
    k: int,
    models_dir: str = "./models",
) -> pd.DataFrame:
    
    df = df.copy()
    df["dominant_cluster"] = np.argmax(probs, axis=1)
    df["dominant_prob"] = np.max(probs, axis=1)

    df["cluster_entropy"] = [scipy_entropy(p) for p in probs]
    df["cluster_probs"] = [p.tolist() for p in probs]
    cluster_summary = []
    for c in range(k):
        mask = df["dominant_cluster"] == c
        cluster_docs = df[mask]
        top_cats = cluster_docs["category"].value_counts().head(5).to_dict()
        cluster_summary.append({
            "cluster": c,
            "size": int(mask.sum()),
            "top_categories": top_cats,
            "mean_entropy": float(df.loc[mask, "cluster_entropy"].mean()),
        })
    
    summary_path = Path(models_dir) / "cluster_summary.json"
    with open(summary_path, "w") as f:
        json.dump(cluster_summary, f, indent=2)

    plot_umap_clusters(reduced_2d, df["dominant_cluster"].values, k, models_dir)
    plot_entropy(df["cluster_entropy"].values, models_dir)

    boundary = df.nlargest(20, "cluster_entropy")[
        ["doc_id", "category", "dominant_cluster", "cluster_entropy", "text"]
    ].copy()
    boundary["text_preview"] = boundary["text"].str[:200]
    boundary_path = Path(models_dir) / "boundary_cases.json"
    boundary.drop(columns=["text"]).to_json(boundary_path, orient="records", indent=2)

    return df

def plot_umap_clusters(
    reduced_2d: np.ndarray,
    labels: np.ndarray,
    k: int,
    models_dir: str,
) -> None:
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab20", k)

    for c in range(k):
        mask = labels == c
        ax.scatter(
            reduced_2d[mask, 0],
            reduced_2d[mask, 1],
            s=1,
            alpha=0.4,
            color=cmap(c),
            label=f"C{c}",
        )
    
    ax.set_title(f"UMAP 2D — {k} GMM clusters (dominant assignment)")
    ax.legend(markerscale=5, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    path = Path(models_dir) / "umap_clusters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_entropy(entropies: np.ndarray, models_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(entropies, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Assignment entropy (higher = more uncertain)")
    ax.set_ylabel("Number of documents")
    ax.set_title("Distribution of cluster assignment uncertainty")
    ax.axvline(np.percentile(entropies, 90), color="red", linestyle="--",
               label="90th percentile (boundary region)")
    ax.legend()
    plt.tight_layout()
    path = Path(models_dir) / "entropy_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

def run_clustering(
    k: Optional[int] = None,
    embeddings_dir: str = "./embeddings",
    models_dir: str = "./models",
    processed_dir: str = "./data/processed",
    k_range: list = [5, 10, 15, 20, 25, 30],
) -> pd.DataFrame:
    
    df = pd.read_parquet(Path(processed_dir) / "corpus.parquet")
    data = np.load(Path(embeddings_dir) / "corpus_embeddings.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    reduced_nd, reduced_2d = reduce_dimensions(
        embeddings, n_components=5, models_dir=models_dir
    )
    if k is None:
        k, bic_scores = select_k_by_bic(reduced_nd, k_range=k_range, models_dir=models_dir)

    gmm, probs = fit_gmm(reduced_nd, k=k, models_dir=models_dir)
    df_clustered = analyse_clusters(df, probs, reduced_2d, k=k, models_dir=models_dir)
    out_path = Path(processed_dir) / "corpus_clustered.parquet"
    save_df = df_clustered.drop(columns=["cluster_probs"])
    save_df.to_parquet(out_path, index=False)

    probs_path = Path(models_dir) / "cluster_probs.npy"
    np.save(probs_path, probs)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from vectorstore import VectorStore
        vs = VectorStore(persist_dir=str(Path(embeddings_dir) / "chromadb"))
        vs.update_cluster_metadata(
            doc_ids=df_clustered["doc_id"].tolist(),
            dominant_clusters=df_clustered["dominant_cluster"].tolist(),
        )
    except Exception as e:
        print(f"{e}")
    
    return df_clustered

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Part 2 fuzzy clustering")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of clusters. If omitted, selected by BIC.")
    parser.add_argument("--embeddings-dir", type=str, default="./embeddings")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--processed-dir", type=str, default="./data/processed")
    args = parser.parse_args()

    run_clustering(
        k=args.k,
        embeddings_dir=args.embeddings_dir,
        models_dir=args.models_dir,
        processed_dir=args.processed_dir,
    )

