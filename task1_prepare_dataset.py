"""
Task 1: Prepare ML-ready dataset (drug, food, interaction)

Input:
    data_with_food_entities.json   # generated from Task 0

Output:
    drug_food_interactions_flat.csv
    drug_food_interactions_flat.json

Goal:
    Convert each row from:
        drug_name | interaction_text | food_entities=[a,b,c]
    into multiple rows:
        drug_name | food=a | interaction_text
        drug_name | food=b | interaction_text
        ...
"""

import json
import pandas as pd

# ============================
# 1. Load Task 0 data
# ============================

INPUT_PATH = (
    "data_with_food_extraction/data_with_food_entities.json"  # update if needed
)

df = pd.read_json(INPUT_PATH)
print(f"Loaded {len(df)} rows from Task 0 output.")


# ============================
# 2. Flatten dataset
# ============================

rows = []  # new expanded rows

for _, row in df.iterrows():
    drug = row["drug_name"]
    interaction = row["interaction_text"]
    food_list = row["food_entities"] or []

    # If no foods detected → still keep a row with food=None?
    # Decision: Keep it, but mark food as None.
    if len(food_list) == 0:
        rows.append({"drug_name": drug, "food": None, "interaction_text": interaction})
        continue

    # Otherwise create one row per food entity
    for food in food_list:
        rows.append({"drug_name": drug, "food": food, "interaction_text": interaction})

df_flat = pd.DataFrame(rows)
print(f"Flattened into {len(df_flat)} (drug, food, interaction) rows.")


# ============================
# 3. Save outputs
# ============================

df_flat.to_csv("drug_food_interactions_flat.csv", index=False)
df_flat.to_json(
    "drug_food_interactions_flat.json", orient="records", force_ascii=False, indent=2
)

print("\nSaved:")
print(" - drug_food_interactions_flat.csv")
print(" - drug_food_interactions_flat.json")

# Show sample
print("\nSample of flattened dataset:")
print(df_flat.head(10))
