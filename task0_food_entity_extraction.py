"""
Task 0: Food / Herb / Supplement Entity Extraction (Hybrid NER with rate limiting)

Behavior:
    - Rule-based + lexicon for ALL rows.
    - Optional Gemini fallback:
        * Only used when rule-based is empty or too generic.
        * Max number of Gemini calls per run is limited.
        * Sleep between calls to avoid per-minute rate-limit spikes.
        * Gracefully handles 429 (quota exceeded) and switches to rule-based only.

Dependencies:
    pip install google-generativeai pandas

Environment (only needed if USE_GEMINI_FALLBACK = True):
    export GEMINI_API_KEY="your_gemini_api_key_here"
"""

import os
import re
import json
import time
from typing import List, Set, Optional

import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


# =========================
# 0. CONFIG
# =========================

# 👉 Turn this ON if you want limited Gemini fallback
USE_GEMINI_FALLBACK = True  # set to False for pure rule-based
MAX_GEMINI_CALLS = 30  # hard cap per run (stays under free-tier daily limits)
SLEEP_BETWEEN_CALLS = 6.0  # seconds between Gemini calls to smooth RPM

GENERATION_MODEL_NAME = "gemini-2.5-flash"  # adjust to a model available to you

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if USE_GEMINI_FALLBACK:
    if GEMINI_API_KEY is None:
        raise ValueError("USE_GEMINI_FALLBACK=True but GEMINI_API_KEY is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
else:
    gemini_model = None  # we won't call Gemini at all


# =========================
# 1. DATA LOADING
# =========================


def load_drug_food_data(json_path: str) -> pd.DataFrame:
    """
    Load the raw JSON data and normalize it into one row per (drug, interaction) pair.

    Expected JSON structure:
        [
          {
            "name": "Lepirudin",
            "reference": "...",
            "food_interactions": ["...", "...", ...]
          },
          ...
        ]

    Returns:
        DataFrame with columns: ["drug_name", "interaction_text", "reference"]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        name = entry.get("name", "").strip()
        reference = entry.get("reference", "").strip()
        interactions = entry.get("food_interactions", []) or []

        for interaction in interactions:
            interaction = (interaction or "").strip()
            if not interaction:
                continue

            records.append(
                {
                    "drug_name": name,
                    "interaction_text": interaction,
                    "reference": reference,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


# =========================
# 2. FOOD LEXICON
# =========================


def load_food_lexicon(path: Optional[str] = None) -> Set[str]:
    """
    Load a curated lexicon of foods, herbs, drinks, and supplements.
    If a file path is given and exists, load additional terms from there (one per line).
    """
    base_terms = [
        # Fruits / juices
        "grapefruit",
        "grapefruit juice",
        "orange juice",
        "cranberry",
        "cranberry juice",
        "apple juice",
        "pomelo",
        "seville orange",
        "fruit juice",
        "citrus juice",
        # Alcohol / beverages
        "alcohol",
        "alcoholic beverages",
        "wine",
        "beer",
        "ethanol",
        "coffee",
        "tea",
        "green tea",
        # General food patterns
        "food",
        "foods",
        "meal",
        "meals",
        "high-fat meal",
        "low-fat meal",
        "dairy",
        "dairy products",
        "milk",
        "yogurt",
        "cheese",
        # Vegetables / vitamin K, etc.
        "spinach",
        "broccoli",
        "kale",
        "leafy green vegetables",
        "leafy greens",
        "vitamin k rich foods",
        "vitamin k-rich foods",
        # Herbs, botanicals, supplements
        "st. john's wort",
        "ginseng",
        "ginkgo",
        "ginkgo biloba",
        "garlic",
        "ginger",
        "echinacea",
        "kava",
        "chamomile",
        "valerian",
        "licorice",
        "milk thistle",
        "saw palmetto",
        "black cohosh",
        "evening primrose oil",
        # Electrolytes and nutrients
        "potassium",
        "potassium rich foods",
        "salt",
        "sodium",
        "vitamin c",
        "vitamin d",
        "vitamin k",
        "iron",
        "calcium",
        # Others commonly seen
        "soy",
        "soy products",
        "soybean",
        "grapefruit products",
        "fiber",
        "dietary fiber",
        "enteral nutrition",
        "tube feeding",
    ]

    lexicon = {t.lower() for t in base_terms}

    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term:
                    lexicon.add(term.lower())

    return lexicon


# =========================
# 3. RULE-BASED EXTRACTION
# =========================

GENERIC_TERMS = {
    "food",
    "foods",
    "meal",
    "meals",
    "fruit",
    "fruits",
    "juice",
    "juices",
    "diet",
    "dietary",
    "alcohol",
    "alcoholic beverages",
}


def normalize_text(text: str) -> str:
    """
    Lowercase and normalize whitespace.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_from_enumerations(text: str) -> List[str]:
    """
    Extract candidate items from patterns like:
        - "such as A, B, and C"
        - "including A and B"
        - "examples include A, B, C"
    """
    norm = normalize_text(text)
    candidates: List[str] = []

    patterns = [
        r"(?:such as|including|include|includes|examples include)\s+([^.;:]+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, norm):
            segment = match.group(1)
            parts = re.split(r",| and | or ", segment)
            for p in parts:
                item = p.strip(" .;:")
                if not item:
                    continue
                if len(item) < 3:
                    continue
                candidates.append(item)

    return candidates


def extract_foods_rule_based(text: str, lexicon: Set[str]) -> List[str]:
    """
    Rule-based food/herb/supplement extraction:
        1. Match lexicon entries present in the text.
        2. Extract enumeration candidates.
    """
    norm = normalize_text(text)
    found: Set[str] = set()

    # Direct lexicon matches
    for term in lexicon:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, norm):
            found.add(term)

    # Enumerations
    enum_items = extract_from_enumerations(norm)
    for item in enum_items:
        found.add(item)

    return sorted(found)


def is_low_information_food_list(foods: List[str]) -> bool:
    """
    True if the list is empty or only contains very generic terms.
    """
    if not foods:
        return True
    lower_foods = {f.lower() for f in foods}
    return lower_foods.issubset(GENERIC_TERMS)


# =========================
# 4. GEMINI HELPERS (WITH RATE LIMIT SAFETY)
# =========================


def safe_parse_food_list(raw: str) -> List[str]:
    """
    Try to parse a JSON list of strings from model output.
    """
    raw = raw.strip()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            data = json.loads(raw)
            return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
        except Exception:
            pass

    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            data = json.loads(snippet)
            return [str(x).strip() for x in data if isinstance(x, (str, int, float))]
        except Exception:
            pass

    return []


def call_gemini_food_extraction(text: str) -> List[str]:
    """
    Use Gemini to extract food/herb/supplement entities.
    Assumes gemini_model is configured globally.
    """
    if gemini_model is None:
        return []

    prompt = f"""
You are an expert in pharmacology and nutrition.

Extract all foods, drinks, herbs, and dietary supplements mentioned in the
following drug–food interaction text. Focus ONLY on the non-drug items:
foods, beverages, herbs, botanicals, and dietary supplements.

Return ONLY a valid JSON list of strings.
Example: ["grapefruit juice", "alcohol", "St. John's wort"]

Text:
\"\"\"{text}\"\"\"
"""
    response = gemini_model.generate_content(prompt)
    raw = (response.text or "").strip()
    foods = safe_parse_food_list(raw)
    return foods


# =========================
# 5. MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # 1) Load data
    # Adjust path if needed (this is where your uploaded file lives in this environment)
    df = load_drug_food_data("data.json")
    print(f"Loaded {len(df)} (drug, interaction) pairs.")

    # 2) Load lexicon
    lexicon_path = None  # or "food_lexicon.txt" if you create one
    lexicon = load_food_lexicon(lexicon_path)
    print(f"Loaded lexicon with {len(lexicon)} terms.")

    # 3) Run extraction
    all_food_entities: List[List[str]] = []
    gemini_calls_used = 0

    for i, row in df.iterrows():
        text = row["interaction_text"]

        # Rule-based extraction first
        rb_foods = extract_foods_rule_based(text, lexicon)

        foods = sorted(set(rb_foods))

        # Gemini fallback only if:
        # - enabled
        # - we still have budget
        # - rule-based result is low-information
        if (
            USE_GEMINI_FALLBACK
            and gemini_calls_used < MAX_GEMINI_CALLS
            and is_low_information_food_list(rb_foods)
        ):
            try:
                llm_foods = call_gemini_food_extraction(text)
                gemini_calls_used += 1
                foods = sorted(set(rb_foods) | set(llm_foods))

                # sleep to respect per-minute rate limits
                if gemini_calls_used < MAX_GEMINI_CALLS:
                    time.sleep(SLEEP_BETWEEN_CALLS)

            except ResourceExhausted as e:
                # Quota exceeded → log once and disable further Gemini calls
                print(
                    "\n[WARNING] Gemini quota exceeded; "
                    "switching to pure rule-based for the rest of this run."
                )
                print("Details:", str(e))
                USE_GEMINI_FALLBACK = False  # avoid more attempts
                # keep 'foods' as rule-based only for this row

        all_food_entities.append(foods)

        # Progress logging
        if (i + 1) % 200 == 0:
            print(
                f"Processed {i+1}/{len(df)} rows..."
                f" | Gemini calls used: {gemini_calls_used}/{MAX_GEMINI_CALLS}"
            )

    df["food_entities"] = all_food_entities
    df["food_entities_str"] = df["food_entities"].apply(lambda xs: "; ".join(xs))

    # 4) Save results
    df.to_csv("data_with_food_entities.csv", index=False)
    df.to_json(
        "data_with_food_entities.json", orient="records", force_ascii=False, indent=2
    )

    print("\nSaved:")
    print("  - data_with_food_entities.csv")
    print("  - data_with_food_entities.json")

    # 5) Show some examples
    print("\nSample rows with extracted food entities:")
    sample = df.head(10)[["drug_name", "interaction_text", "food_entities_str"]]
    for _, row in sample.iterrows():
        print("Drug:", row["drug_name"])
        print("Interaction:", row["interaction_text"])
        print("Foods:", row["food_entities_str"])
        print("-" * 60)

    print(f"\nRun completed. Gemini calls used: {gemini_calls_used}/{MAX_GEMINI_CALLS}")
