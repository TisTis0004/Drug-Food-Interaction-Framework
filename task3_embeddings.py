"""
Task 3: Sentence Embeddings for (drug, food, interaction) triples

Model:
    sentence-transformers: all-mpnet-base-v2

Input:
    /mnt/data/drug_food_interactions_labeled.json

Outputs:
    drug_food_interactions_embedded.csv        # same rows (no huge vectors inside)
    embeddings_all_mpnet_base_v2.npy           # NumPy array of shape (N, 768)
    metadata_embeddings_mapping.json           # simple info about model & shapes
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


# ==========================
# 1. Config
# ==========================

INPUT_PATH = "data_severity_labeled/drug_food_interactions_labeled.json"

# Name of the sentence-transformers model
ST_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Output paths
OUTPUT_CSV = "drug_food_interactions_embedded.csv"
OUTPUT_NPY = "embeddings_all_mpnet_base_v2.npy"
OUTPUT_META = "metadata_embeddings_mapping.json"


# ==========================
# 2. Load labeled dataset
# ==========================

print(f"Loading labeled dataset from: {INPUT_PATH}")
df = pd.read_json(INPUT_PATH)
print(f"Loaded {len(df)} labeled rows.")


# ==========================
# 3. Build text representation
# ==========================


def build_embedding_text(row: pd.Series) -> str:
    """
    Construct a canonical text for embedding that includes drug, food, and interaction context.
    Example:
        "Drug: Warfarin. Food: leafy green vegetables. Interaction: Vitamin K rich foods may reduce..."
    """
    drug = str(row.get("drug_name", "")).strip()
    food = row.get("food", None)
    if food is None or (isinstance(food, float) and np.isnan(food)):
        food_str = "general food"
    else:
        food_str = str(food).strip()

    interaction = str(row.get("interaction_text", "")).strip()

    text = f"Drug: {drug}. Food: {food_str}. Interaction: {interaction}"
    return text


texts = [build_embedding_text(row) for _, row in df.iterrows()]
print(f"Prepared {len(texts)} texts for embedding.")


# ==========================
# 4. Load embedding model
# ==========================

print(f"Loading sentence-transformers model: {ST_MODEL_NAME}")
model = SentenceTransformer(ST_MODEL_NAME)


# ==========================
# 5. Encode in batches
# ==========================

# You can adjust batch_size depending on your RAM/GPU
BATCH_SIZE = 64

print("Encoding texts into embeddings...")
embeddings_list = []
for start in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[start : start + BATCH_SIZE]
    batch_embs = model.encode(
        batch_texts,
        batch_size=len(batch_texts),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity friendly
    )
    embeddings_list.append(batch_embs)

embeddings = np.vstack(embeddings_list)
print(f"Embeddings shape: {embeddings.shape}")  # (N, 768)


# ==========================
# 6. Save outputs
# ==========================

# 6.1 Save embeddings separately as .npy (efficient)
np.save(OUTPUT_NPY, embeddings)
print(f"Saved embeddings to: {OUTPUT_NPY}")

# 6.2 Save the dataframe (no huge vectors inside → just the original columns)
# You can also add an index column if you want explicit mapping
df = df.reset_index(drop=True)
df["row_id"] = df.index  # explicit id to align with embeddings row index

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved metadata (without vectors) to: {OUTPUT_CSV}")

# 6.3 Save some metadata about the embedding mapping
meta = {
    "input_file": INPUT_PATH,
    "output_csv": OUTPUT_CSV,
    "output_npy": OUTPUT_NPY,
    "model_name": ST_MODEL_NAME,
    "num_rows": int(df.shape[0]),
    "embedding_dim": int(embeddings.shape[1]),
    "normalized": True,
    "text_format": "Drug: {drug_name}. Food: {food or 'general food'}. Interaction: {interaction_text}",
}
with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"Saved embedding metadata to: {OUTPUT_META}")

# ==========================
# 7. Quick sanity check
# ==========================

print("\nSample rows:")
print(df.head(5)[["row_id", "drug_name", "food", "interaction_text", "severity"]])

print("\nDone. You can now load embeddings like:")
print("  df = pd.read_csv('drug_food_interactions_embedded.csv')")
print("  embs = np.load('embeddings_all_mpnet_base_v2.npy')")
print("  # row i in df corresponds to embs[i]")
