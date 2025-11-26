"""
Task 2: Severity Labeling for (drug, food, interaction) triples

Input:
    drug_food_interactions_flat.json

Output:
    drug_food_interactions_labeled.csv
    drug_food_interactions_labeled.json

Pipeline:
    1. Rule-based classifier for severity (severe, moderate, minor, info)
    2. Optional Gemini fallback (limited calls + rate limiting)
    3. Final label = rule-based OR rule + llm
"""

import json
import os
import re
import time
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


# ========================
# Config
# ========================

INPUT_PATH = "data_food_flat/drug_food_interactions_flat.json"

USE_GEMINI_FALLBACK = True  # Turn ON to help classify ambiguous cases
MAX_GEMINI_CALLS = 40  # Hard cap to avoid quota errors
SLEEP_BETWEEN_CALLS = 6.0  # Seconds between Gemini calls
GENERATION_MODEL_NAME = "gemini-2.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # <-- same as Task 0

if USE_GEMINI_FALLBACK:
    if GEMINI_API_KEY is None:
        raise ValueError(
            "USE_GEMINI_FALLBACK=True but GEMINI_API_KEY is not set in environment variables."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
else:
    gemini_model = None


# ========================
# Rule-based Patterns
# ========================

SEVERE_PATTERNS = [
    r"life[- ]?threatening",
    r"fatal",
    r"contraindicated",
    r"rhabdomyolysis",
    r"severe bleeding",
    r"major bleeding",
    r"hemorrhage",
    r"dangerous",
    r"avoid grapefruit",  # many statins
]

MODERATE_PATTERNS = [
    r"increase(s|d)? (the )?level",
    r"may increase",
    r"raise(s|d)?",
    r"reduce absorption",
    r"may decrease",
    r"dose adjustment",
    r"monitor",
]

MINOR_PATTERNS = [
    r"take with food",
    r"food increases absorption",
    r"may cause stomach upset",
    r"nausea",
    r"drink plenty of fluids",
]

INFO_PATTERNS = [
    r"no significant effect",
    r"minimal effect",
    r"not clinically important",
]


def classify_severity_rule(text: str) -> str:
    """
    Rule-based severity classifier.
    Returns one of: severe, moderate, minor, info, unknown
    """

    text_l = text.lower()

    for p in SEVERE_PATTERNS:
        if re.search(p, text_l):
            return "severe"

    for p in MODERATE_PATTERNS:
        if re.search(p, text_l):
            return "moderate"

    for p in MINOR_PATTERNS:
        if re.search(p, text_l):
            return "minor"

    for p in INFO_PATTERNS:
        if re.search(p, text_l):
            return "info"

    return "unknown"


# ============================
# Gemini fallback classifier
# ============================


def classify_with_gemini(text: str) -> str:
    """
    Calls Gemini to classify severity.
    Returns: severe, moderate, minor, info
    """

    prompt = f"""
You are a pharmacology expert.

Classify the SEVERITY of this food interaction as one of:
"severe", "moderate", "minor", or "info".

Severe = life-threatening, high harm, major interaction
Moderate = clinically relevant and requires caution
Minor = mild effect or common precaution
Info = minimal or no clinically meaningful concern

Return ONLY a single word: severe/moderate/minor/info

Interaction text:
{text}
"""

    response = gemini_model.generate_content(prompt)
    answer = (response.text or "").strip().lower()

    if "severe" in answer:
        return "severe"
    if "moderate" in answer:
        return "moderate"
    if "minor" in answer:
        return "minor"

    return "info"  # default safe class


# ============================
# Main Pipeline
# ============================

df = pd.read_json(INPUT_PATH)
print(f"Loaded {len(df)} flattened interaction rows.")

labels = []
gemini_used = 0

for i, row in df.iterrows():
    text = row["interaction_text"]

    # 1. Rule-based first
    rule_label = classify_severity_rule(text)

    if rule_label != "unknown":
        labels.append(rule_label)
        continue

    # 2. Gemini fallback (only if enabled and under quota)
    if USE_GEMINI_FALLBACK and gemini_used < MAX_GEMINI_CALLS:
        try:
            llm_label = classify_with_gemini(text)
            labels.append(llm_label)
            gemini_used += 1
            time.sleep(SLEEP_BETWEEN_CALLS)
        except ResourceExhausted:
            print("\n[WARN] Gemini quota exceeded. Continuing with rule-based only.")
            USE_GEMINI_FALLBACK = False
            labels.append("info")
    else:
        labels.append("info")

df["severity"] = labels


# ============================
# Save Outputs
# ============================

df.to_csv("drug_food_interactions_labeled.csv", index=False)
df.to_json(
    "drug_food_interactions_labeled.json", orient="records", force_ascii=False, indent=2
)

print("\nSaved:")
print(" - drug_food_interactions_labeled.csv")
print(" - drug_food_interactions_labeled.json")
print(f"\nGemini calls used: {gemini_used}/{MAX_GEMINI_CALLS}")

print("\nSample labeled rows:")
print(df.head(10))
