"""
Task 7: Zero-Shot / Similarity-Based Severity Prediction

Goal:
    Use the similarity engine + embeddings to estimate the severity of
    NEW (drug, food, interaction_text) triples, even if they were not
    seen during training.

Inputs (from previous tasks, in data_embeddings/):
    - embeddings_all_mpnet_base_v2.npy
    - drug_food_interactions_clustered.csv
    - similarity_index_nn.pkl
    - best_severity_classifier.pkl
    - label_encoder_severity.pkl

Outputs:
    - No new files required (this is a utility / demo script),
      but you can re-use the functions in your app / agents.

Behavior:
    - Build canonical text representation.
    - Embed query with all-mpnet-base-v2.
    - Retrieve top-k neighbors using NearestNeighbors (cosine).
    - Compute weighted vote over neighbor severities.
    - Optionally, compare with classifier prediction (MLP).
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

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

BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_severity_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_severity.pkl")

ST_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_K = 10  # number of neighbors for zero-shot prediction


# ==========================
# 2. Load artifacts
# ==========================

print(f"Loading embeddings from: {EMB_PATH}")
embeddings = np.load(EMB_PATH)
print(f"Embeddings shape: {embeddings.shape}")

print(f"Loading clustered DataFrame from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"DataFrame shape: {df.shape}")
print("Columns:", df.columns.tolist())

assert (
    embeddings.shape[0] == df.shape[0]
), "Mismatch between embeddings and dataframe rows!"

print(f"Loading NearestNeighbors index from: {INDEX_PATH}")
nn_model: NearestNeighbors = joblib.load(INDEX_PATH)
print("NearestNeighbors index loaded.")

print(f"Loading best severity classifier from: {BEST_MODEL_PATH}")
clf = joblib.load(BEST_MODEL_PATH)
print("Classifier loaded.")

print(f"Loading severity label encoder from: {LABEL_ENCODER_PATH}")
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("Label encoder loaded.")

if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("Similarity index meta:", meta)
else:
    meta = None


# ==========================
# 3. Embedding model for queries
# ==========================

print(f"\nLoading embedding model for queries: {ST_MODEL_NAME}")
st_model = SentenceTransformer(ST_MODEL_NAME)
print("Embedding model loaded.")


def build_embedding_text(drug_name: str, food, interaction_text: str) -> str:
    """
    Construct canonical text, consistent with Task 3.
    """
    drug = str(drug_name).strip()

    if food is None or (isinstance(food, float) and np.isnan(food)):
        food_str = "general food"
    else:
        food_str = str(food).strip()

    interaction = str(interaction_text).strip()

    return f"Drug: {drug}. Food: {food_str}. Interaction: {interaction}"


def embed_query_text(text: str) -> np.ndarray:
    """
    Embed a single text into a normalized MPNet embedding (1, dim).
    """
    emb = st_model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb  # shape (1, dim)


# ==========================
# 4. Zero-shot severity estimation via neighbors
# ==========================


def zero_shot_severity(drug_name: str, food, interaction_text: str, k: int = DEFAULT_K):
    """
    Estimate severity based on nearest neighbors in embedding space.

    Steps:
        - Build canonical text
        - Embed query
        - Retrieve top-k neighbors
        - Convert distances -> similarities
        - Weighted vote over neighbor severities
    Returns:
        dict with:
            - "pred_severity_neighbor"
            - "neighbor_distribution"
            - "neighbor_examples" (top neighbors)
    """

    text = build_embedding_text(drug_name, food, interaction_text)
    query_emb = embed_query_text(text)  # (1, dim)

    distances, indices = nn_model.kneighbors(query_emb, n_neighbors=k)
    distances = distances[0]
    indices = indices[0]

    # Convert cosine distance -> similarity: sim = 1 - dist
    sims = 1.0 - distances
    sims = np.clip(sims, 0.0, 1.0)

    # Aggregate weights by severity label
    severity_weights = {}
    neighbor_rows = []

    for idx, dist, sim in zip(indices, distances, sims):
        row = df.iloc[idx]
        sev = str(row["severity"])
        severity_weights.setdefault(sev, 0.0)
        severity_weights[sev] += float(sim)

        neighbor_rows.append(
            {
                "row_id": int(idx),
                "drug_name": row["drug_name"],
                "food": row["food"],
                "severity": sev,
                "interaction_text": row["interaction_text"],
                "distance": float(dist),
                "similarity": float(sim),
            }
        )

    # Normalize weights to probabilities
    total_weight = sum(severity_weights.values()) or 1.0
    severity_probs = {sev: w / total_weight for sev, w in severity_weights.items()}

    # Choose severity with highest weight
    pred_sev = max(severity_probs.items(), key=lambda x: x[1])[0]
    confidence = severity_probs[pred_sev]

    result = {
        "query_text": text,
        "pred_severity_neighbor": pred_sev,
        "neighbor_severity_distribution": severity_probs,
        "neighbor_confidence": confidence,
        "neighbors": neighbor_rows,
    }

    return result


# ==========================
# 5. Classifier-based prediction (for comparison)
# ==========================


def classifier_severity(drug_name: str, food, interaction_text: str):
    """
    Predict severity using the trained MLP classifier.
    """
    text = build_embedding_text(drug_name, food, interaction_text)
    query_emb = embed_query_text(text)  # (1, dim)

    # classifier expects same embedding format as training
    # (we used normalized MPNet embeddings)
    y_proba = clf.predict_proba(query_emb)[0]
    y_idx = int(np.argmax(y_proba))
    pred_label = label_encoder.inverse_transform([y_idx])[0]

    prob_dict = {
        label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(y_proba)
    }

    return {
        "query_text": text,
        "pred_severity_classifier": str(pred_label),
        "classifier_probabilities": prob_dict,
    }


# ==========================
# 6. Demo
# ==========================


def print_zero_shot_demo(
    drug_name: str, food, interaction_text: str, k: int = DEFAULT_K
):
    print("\n" + "=" * 80)
    print("ZERO-SHOT SEVERITY DEMO")
    print("=" * 80)
    print(f"Drug: {drug_name}")
    print(f"Food: {food}")
    print(f"Interaction: {interaction_text}")
    print("-" * 80)

    # Neighbor-based
    nz = zero_shot_severity(drug_name, food, interaction_text, k=k)
    print(f"Neighbor-based predicted severity: {nz['pred_severity_neighbor']}")
    print(f"Neighbor-based confidence: {nz['neighbor_confidence']:.3f}")
    print("Neighbor-based severity distribution:")
    for sev, prob in nz["neighbor_severity_distribution"].items():
        print(f"  {sev}: {prob:.3f}")

    print("\nTop neighbors:")
    for i, nbr in enumerate(nz["neighbors"], start=1):
        print(
            f"{i}. row_id={nbr['row_id']} | dist={nbr['distance']:.4f} | sim={nbr['similarity']:.4f}"
        )
        print(f"   Drug: {nbr['drug_name']}")
        print(f"   Food: {nbr['food']}")
        print(f"   Severity: {nbr['severity']}")
        print(f"   Interaction: {nbr['interaction_text']}")
        print("-" * 60)

    # Classifier-based
    cz = classifier_severity(drug_name, food, interaction_text)
    print("\nClassifier-based predicted severity:", cz["pred_severity_classifier"])
    print("Classifier probability distribution:")
    for sev, prob in cz["classifier_probabilities"].items():
        print(f"  {sev}: {prob:.3f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example 1: Atorvastatin + grapefruit juice
    example_drug_1 = "Atorvastatin"
    example_food_1 = "grapefruit juice"
    example_interaction_1 = "Avoid grapefruit products. Grapefruit may increase serum levels of atorvastatin and risk of myopathy."
    print_zero_shot_demo(example_drug_1, example_food_1, example_interaction_1, k=10)

    # Example 2: Warfarin + leafy green vegetables
    example_drug_2 = "Warfarin"
    example_food_2 = "leafy green vegetables"
    example_interaction_2 = "High vitamin K intake from leafy green vegetables may reduce the anticoagulant effect of warfarin."
    print_zero_shot_demo(example_drug_2, example_food_2, example_interaction_2, k=10)

    # Example 3: Random supplement + mild effect
    example_drug_3 = "Metformin"
    example_food_3 = "food"
    example_interaction_3 = "Take with food to reduce gastrointestinal side effects."
    print_zero_shot_demo(example_drug_3, example_food_3, example_interaction_3, k=10)

    print("Task 7: zero-shot severity demos completed.")
