"""
Task 5: Clustering of Drug–Food Interactions using UMAP + KMeans

Inputs:
    - data_embeddings/embeddings_all_mpnet_base_v2.npy
    - data_embeddings/drug_food_interactions_embedded.csv

Outputs (in data_embeddings/):
    - data_embeddings/drug_food_interactions_clustered.csv
    - data_embeddings/umap_2d_embeddings.npy
    - data_embeddings/clustering_summary.json
    - data_embeddings/umap_severity.png
    - data_embeddings/umap_kmeans_clusters.png
    - data_embeddings/umap_hdbscan_clusters.png  (if HDBSCAN available)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import umap

# HDBSCAN is optional
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[INFO] hdbscan not installed. HDBSCAN clustering will be skipped.")


# ==========================
# 1. Paths / Config
# ==========================

DATA_DIR = "data_embeddings"

EMB_PATH = os.path.join(DATA_DIR, "embeddings_all_mpnet_base_v2.npy")
CSV_PATH = os.path.join(DATA_DIR, "drug_food_interactions_embedded.csv")

CLUSTERED_CSV = os.path.join(DATA_DIR, "drug_food_interactions_clustered.csv")
UMAP_NPY = os.path.join(DATA_DIR, "umap_2d_embeddings.npy")
SUMMARY_JSON = os.path.join(DATA_DIR, "clustering_summary.json")

PLOT_SEVERITY = os.path.join(DATA_DIR, "umap_severity.png")
PLOT_KMEANS = os.path.join(DATA_DIR, "umap_kmeans_clusters.png")
PLOT_HDBSCAN = os.path.join(DATA_DIR, "umap_hdbscan_clusters.png")

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "cosine"

# KMeans parameters
KMEANS_N_CLUSTERS = 10
KMEANS_RANDOM_STATE = 42


# ==========================
# 2. Load data
# ==========================

print(f"Loading embeddings from: {EMB_PATH}")
embeddings = np.load(EMB_PATH)
print(f"Embeddings shape: {embeddings.shape}")

print(f"Loading metadata CSV from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"DataFrame shape: {df.shape}")
print("Columns in DataFrame:", df.columns.tolist())

assert (
    embeddings.shape[0] == df.shape[0]
), "Mismatch between embeddings and DataFrame rows!"


# ==========================
# 3. UMAP (2D) projection
# ==========================

print("\nRunning UMAP (2D) dimensionality reduction...")

umap_model = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=UMAP_N_COMPONENTS,
    metric=UMAP_METRIC,
    random_state=42,
)

umap_2d = umap_model.fit_transform(embeddings)
print(f"UMAP 2D shape: {umap_2d.shape}")

np.save(UMAP_NPY, umap_2d)
print(f"Saved UMAP 2D embeddings to: {UMAP_NPY}")

df["umap_2d_x"] = umap_2d[:, 0]
df["umap_2d_y"] = umap_2d[:, 1]


# ==========================
# 4. KMeans clustering
# ==========================

print(f"\nRunning KMeans with k={KMEANS_N_CLUSTERS}...")

kmeans = KMeans(
    n_clusters=KMEANS_N_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init="auto"
)

kmeans_labels = kmeans.fit_predict(embeddings)
df["cluster_kmeans"] = kmeans_labels

kmeans_cluster_counts = df["cluster_kmeans"].value_counts().sort_index().to_dict()

print("KMeans cluster sizes:", kmeans_cluster_counts)


# ==========================
# 5. Optional HDBSCAN clustering
# ==========================

if HDBSCAN_AVAILABLE:
    print("\nRunning HDBSCAN clustering on UMAP space...")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=30, metric="euclidean", cluster_selection_method="eom"
    )
    hdb_labels = hdb.fit_predict(umap_2d)
    df["cluster_hdbscan"] = hdb_labels

    hdb_counts = df["cluster_hdbscan"].value_counts().sort_index().to_dict()
    print("HDBSCAN cluster sizes:", hdb_counts)
else:
    print("\n[INFO] Skipping HDBSCAN clustering (library not available).")
    df["cluster_hdbscan"] = -1  # placeholder


# ==========================
# 6. Plotting helpers (fixed)
# ==========================


def plot_umap_categorical(df_plot, label_col, title, out_path, cmap="tab10"):
    """
    Plot UMAP with a categorical column (e.g., severity, cluster labels).
    Converts categories to numeric codes internally, then builds a legend.
    """
    labels = df_plot[label_col].astype("category")
    codes = labels.cat.codes  # integer codes 0..K-1

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df_plot["umap_2d_x"], df_plot["umap_2d_y"], c=codes, cmap=cmap, alpha=0.7, s=10
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # Build legend
    unique_labels = labels.cat.categories
    handles = []
    for i, lab in enumerate(unique_labels):
        handles.append(plt.Line2D([], [], marker="o", linestyle="", label=str(lab)))
    plt.legend(
        handles=handles, title=label_col, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_umap_numeric(df_plot, value_col, title, out_path, cmap="viridis"):
    """
    Plot UMAP with a numeric column (e.g., cluster ids).
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df_plot["umap_2d_x"],
        df_plot["umap_2d_y"],
        c=df_plot[value_col],
        cmap=cmap,
        alpha=0.7,
        s=10,
    )
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter, label=value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot: {out_path}")


# ==========================
# 7. Generate plots
# ==========================

# UMAP by severity (categorical)
if "severity" in df.columns:
    plot_umap_categorical(
        df_plot=df,
        label_col="severity",
        title="UMAP of Drug–Food Interactions (colored by severity)",
        out_path=PLOT_SEVERITY,
    )

# UMAP by KMeans cluster (numeric)
plot_umap_numeric(
    df_plot=df,
    value_col="cluster_kmeans",
    title="UMAP of Drug–Food Interactions (colored by KMeans clusters)",
    out_path=PLOT_KMEANS,
)

# UMAP by HDBSCAN cluster (numeric, if available)
if HDBSCAN_AVAILABLE:
    plot_umap_numeric(
        df_plot=df,
        value_col="cluster_hdbscan",
        title="UMAP of Drug–Food Interactions (colored by HDBSCAN clusters)",
        out_path=PLOT_HDBSCAN,
    )


# ==========================
# 8. Save clustered DataFrame + summary
# ==========================

df.to_csv(CLUSTERED_CSV, index=False)
print(f"\nSaved clustered dataset to: {CLUSTERED_CSV}")

summary = {
    "embeddings_path": EMB_PATH,
    "csv_input_path": CSV_PATH,
    "csv_output_path": CLUSTERED_CSV,
    "umap_n_neighbors": UMAP_N_NEIGHBORS,
    "umap_min_dist": UMAP_MIN_DIST,
    "umap_metric": UMAP_METRIC,
    "kmeans_n_clusters": KMEANS_N_CLUSTERS,
    "kmeans_cluster_sizes": kmeans_cluster_counts,
    "hdbscan_available": HDBSCAN_AVAILABLE,
}

if HDBSCAN_AVAILABLE:
    summary["hdbscan_cluster_sizes"] = {str(k): int(v) for k, v in hdb_counts.items()}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Saved clustering summary to: {SUMMARY_JSON}")
print("\nTask 5 (UMAP + KMeans) completed.")
