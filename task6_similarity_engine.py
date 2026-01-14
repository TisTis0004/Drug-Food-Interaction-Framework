"""
Task 6: Similarity Engine (Nearest Neighbor Retrieval)

Goal:
    Build a k-NN similarity index over the MPNet embeddings so we can:
        - Find similar drug–food interactions.
        - Support zero-shot severity / mechanism reasoning.
        - Power retrieval for the multi-agent system and Streamlit app.

Inputs (from previous tasks in data_embeddings/):
    - embeddings_all_mpnet_base_v2.npy
    - drug_food_interactions_clustered.csv

Outputs:
    - data_embeddings/similarity_index_nn.pkl
    - data_embeddings/similarity_index_meta.json

The script also includes small demo functions for:
    - Query by existing row_id
    - Query by new text (embedding with all-mpnet-base-v2)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


# ==========================
# 1. Paths / Config
# ==========================

DATA_DIR = "data_embeddings"

EMB_PATH = os.path.join(DATA_DIR, "embeddings_all_mpnet_base_v2.npy")
CSV_PATH = os.path.join(DATA_DIR, "drug_food_interactions_clustered.csv")

INDEX_PATH = os.path.join(DATA_DIR, "similarity_index_nn.pkl")
META_PATH = os.path.join(DATA_DIR, "similarity_index_meta.json")

# Same embedding model we used in Task 3
ST_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Default number of neighbors to retrieve in demos
DEFAULT_K = 5


# ==========================
# 2. Load embeddings + metadata
# ==========================

print(f"Loading embeddings from: {EMB_PATH}")
embeddings = np.load(EMB_PATH)
print(f"Embeddings shape: {embeddings.shape}")  # (N, 768)

print(f"Loading DataFrame from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"DataFrame shape: {df.shape}")
print("Columns:", df.columns.tolist())

assert embeddings.shape[0] == df.shape[0], "Mismatch between rows and embeddings!"

# Ensure a stable row_id column exists and matches embedding order
if "row_id" not in df.columns:
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

# Save back (optional) to ensure row_id is always there
df.to_csv(CSV_PATH, index=False)


# ==========================
# 3. Build NearestNeighbors index
# ==========================

print("\nBuilding NearestNeighbors index (cosine distance)...")

# We normalize embeddings in Task 3, so cosine distance is fine
nn_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",  # robust for high-dimensional embeddings
)

nn_model.fit(embeddings)
print("NearestNeighbors index built.")

# Save index
joblib.dump(nn_model, INDEX_PATH)
print(f"Saved nearest neighbor index to: {INDEX_PATH}")


# ==========================
# 4. Save meta info
# ==========================

meta = {
    "embeddings_path": EMB_PATH,
    "csv_path": CSV_PATH,
    "index_path": INDEX_PATH,
    "model_name": ST_MODEL_NAME,
    "num_rows": int(embeddings.shape[0]),
    "embedding_dim": int(embeddings.shape[1]),
    "metric": "cosine",
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"Saved similarity index metadata to: {META_PATH}")


# ==========================
# 5. Helper: build text for embedding
# ==========================


def build_embedding_text(drug_name: str, food, interaction_text: str) -> str:
    """
    Construct the canonical text representation for embedding, consistent with Task 3.
    Example:
        "Drug: Warfarin. Food: leafy green vegetables. Interaction: Vitamin K rich foods may reduce..."
    """
    drug = str(drug_name).strip()

    if food is None or (isinstance(food, float) and np.isnan(food)):
        food_str = "general food"
    else:
        food_str = str(food).strip()

    interaction = str(interaction_text).strip()

    return f"Drug: {drug}. Food: {food_str}. Interaction: {interaction}"


# ==========================
# 6. Helper: load embedding model for new queries
# ==========================

print(f"\nLoading sentence-transformers model for query embedding: {ST_MODEL_NAME}")
st_model = SentenceTransformer(ST_MODEL_NAME)
print("Model loaded.")


def embed_texts(texts):
    """
    Embed a list of texts using the same MPNet model and normalization as before.
    """
    embs = st_model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )
    return embs


# ==========================
# 7. Retrieval functions (for demonstration)
# ==========================


def retrieve_by_row_id(
    row_id: int,
    k: int = DEFAULT_K,
):
    """
    Retrieve the top-k most similar interactions, given an existing row_id.
    """
    if row_id < 0 or row_id >= embeddings.shape[0]:
        raise ValueError(
            f"row_id {row_id} is out of range [0, {embeddings.shape[0] - 1}]"
        )

    query_emb = embeddings[row_id : row_id + 1]  # shape (1, dim)

    distances, indices = nn_model.kneighbors(query_emb, n_neighbors=k)
    distances = distances[0]
    indices = indices[0]

    print(f"\nQuery row_id: {row_id}")
    print("Query:")
    qrow = df.iloc[row_id]
    print(f"  Drug: {qrow['drug_name']}")
    print(f"  Food: {qrow['food']}")
    print(f"  Severity: {qrow['severity']}")
    print(f"  Interaction: {qrow['interaction_text']}\n")

    print(f"Top-{k} nearest neighbors (including the query itself at distance 0):")
    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df.iloc[idx]
        print(f"{rank}. row_id={idx} | cosine distance={dist:.4f}")
        print(f"   Drug: {row['drug_name']}")
        print(f"   Food: {row['food']}")
        print(f"   Severity: {row['severity']}")
        print(f"   Interaction: {row['interaction_text']}")
        print("-" * 80)


def retrieve_by_free_text(
    drug_name: str,
    food,
    interaction_text: str,
    k: int = DEFAULT_K,
):
    """
    Free-text query: build the canonical text, embed it, and retrieve nearest neighbors.
    This can be used for:
        - New (drug, food, interaction) triples
        - Hypothetical interactions
        - User queries
    """
    text = build_embedding_text(drug_name, food, interaction_text)
    query_emb = embed_texts([text])  # shape (1, dim)

    distances, indices = nn_model.kneighbors(query_emb, n_neighbors=k)
    distances = distances[0]
    indices = indices[0]

    print("\nFree-text query:")
    print(f"  {text}\n")
    print(f"Top-{k} nearest neighbors:")

    for rank, (idx, dist) in enumerate(zip(indices, distances), start=1):
        row = df.iloc[idx]
        print(f"{rank}. row_id={idx} | cosine distance={dist:.4f}")
        print(f"   Drug: {row['drug_name']}")
        print(f"   Food: {row['food']}")
        print(f"   Severity: {row['severity']}")
        print(f"   Interaction: {row['interaction_text']}")
        print("-" * 80)


# ==========================
# 8. Demo usage (optional)
# ==========================

if __name__ == "__main__":
    # Example 1: use an existing row as query
    example_row_id = 0
    retrieve_by_row_id(example_row_id, k=5)

    # Example 2: free-text query for a hypothetical interaction
    example_drug = "Atorvastatin"
    example_food = "grapefruit juice"
    example_interaction = (
        "Avoid grapefruit products. Grapefruit may increase serum levels."
    )
    retrieve_by_free_text(example_drug, example_food, example_interaction, k=5)

    print("\nTask 6: similarity index built and demo retrievals completed.")
