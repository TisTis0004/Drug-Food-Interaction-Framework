"""
Task 8: Multi-Agent Explainability with Arabic Output + Relevance Checker

Agents:
    0) Relevance Checker Agent        (local)
    1) Retrieval Agent                (local)
    2) Severity Aggregation Agent     (local)
    3) Clinical Reasoning Agent       (local)
    4) Arabic Explanation Agent       (Gemini first, local LLM fallback)
    5) Safety Checker Agent           (local)

Inputs (from previous tasks in data_embeddings/):
    - embeddings_all_mpnet_base_v2.npy
    - drug_food_interactions_clustered.csv
    - similarity_index_nn.pkl
    - best_severity_classifier.pkl
    - label_encoder_severity.pkl

Usage:
    python task8_multi_agent_explainer.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Try to import Gemini (google-generativeai)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Transformers for local Arabic fallback
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==========================
# 1. Paths / Global Config
# ==========================

DATA_DIR = "data_embeddings"

EMB_PATH = os.path.join(DATA_DIR, "embeddings_all_mpnet_base_v2.npy")
CSV_PATH = os.path.join(DATA_DIR, "drug_food_interactions_clustered.csv")
INDEX_PATH = os.path.join(DATA_DIR, "similarity_index_nn.pkl")
BEST_MODEL_PATH = os.path.join(DATA_DIR, "best_severity_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder_severity.pkl")
META_PATH = os.path.join(DATA_DIR, "similarity_index_meta.json")

ST_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Gemini config (Arabic Explanation Agent primary)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # adjust if needed

# Local Arabic fallback model (multilingual generative)
LOCAL_AR_MODEL_NAME = "google/mt5-small"  # small, multilingual, can output Arabic

# Number of neighbors for retrieval / zero-shot
K_NEIGHBORS = 10


# ==========================
# 2. Load Core Artifacts
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
        similarity_meta = json.load(f)
    print("Similarity index meta:", similarity_meta)
else:
    similarity_meta = None

print(f"\nLoading embedding model for queries: {ST_MODEL_NAME}")
st_model = SentenceTransformer(ST_MODEL_NAME)
print("Embedding model loaded.")

# Drug and food vocabularies (for relevance checking)
DRUG_VOCAB = set(df["drug_name"].dropna().astype(str).str.lower().unique())
FOOD_VOCAB = set(df["food"].dropna().astype(str).str.lower().unique())

# Build a semantic index for foods/herbs (for fuzzy matching / "translation")
FOOD_LIST = sorted(df["food"].dropna().astype(str).unique())
print(f"Unique foods/herbs in dataset: {len(FOOD_LIST)}")
print("Encoding food/herb names into embeddings for semantic matching...")
FOOD_EMBS = st_model.encode(
    FOOD_LIST,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print("Food embeddings shape:", FOOD_EMBS.shape)


# ================
# Injectable drugs
# ================


def build_injectable_drug_set(df: pd.DataFrame):
    """
    Hybrid approach:
      - Start from a curated list of known IV/SC/IM agents.
      - Add drugs that show IV/injection keywords in interaction_text
        and lack clear oral/food instructions.
    """
    # Curated list (lower-case). You can extend this over time.
    curated_injectables = {
        "alteplase",
        "reteplase",
        "tenecteplase",
        "urokinase",
        "streptokinase",
        "abciximab",
        "eptifibatide",
        "tirofiban",
        "heparin",
        "enoxaparin",
        "dalteparin",
        "fondaparinux",
        "darbepoetin alfa",
        "erythropoietin",
        "epoetin alfa",
        "peginterferon alfa-2a",
        "peginterferon alfa-2b",
        "interferon alfa-n1",
        "interferon alfa-n3",
        "interferon beta-1a",
        "interferon beta-1b",
        "interferon gamma-1b",
    }

    iv_tokens = [
        "intravenous",
        " iv ",
        "iv infusion",
        "infusion",
        "bolus",
        "injection",
        "inject",
        "subcutaneous",
        "intramuscular",
        "im injection",
        "sc injection",
    ]

    oral_tokens = [
        "take with food",
        "take without regard to food",
        "take before meals",
        "take after meals",
        "take with or without food",
        "orally",
        "by mouth",
        "tablet",
        "capsule",
        "swallow",
        "food increases bioavailability",
        "food decreases bioavailability",
    ]

    auto_injectables = set()

    for drug in df["drug_name"].dropna().astype(str).unique():
        sub = df[df["drug_name"] == drug]["interaction_text"].dropna().astype(str)
        txt = " ".join(sub).lower()
        if not txt:
            continue

        has_iv = any(tok in txt for tok in iv_tokens)
        has_oral = any(tok in txt for tok in oral_tokens)

        # Heuristic: IV mentions and no clear oral/food intake wording
        if has_iv and not has_oral:
            auto_injectables.add(drug.lower())

    hybrid_set = curated_injectables.union(auto_injectables)
    return hybrid_set


INJECTABLE_DRUGS = build_injectable_drug_set(df)
print(f"Injectable / non-oral drugs detected (hybrid): {len(INJECTABLE_DRUGS)}")


# Some obviously non-medical / non-food keywords for quick filtering
IRRELEVANT_KEYWORDS = {
    "windows",
    "android",
    "iphone",
    "laptop",
    "computer",
    "pc",
    "playstation",
    "xbox",
    "television",
    "car",
    "engine",
    "router",
    "wifi",
    "internet",
    "server",
    "linux",
    "macos",
}

# Herbs / foods with anticoagulant or antiplatelet traits (for bleeding risk)
ANTICOAGULANT_FOODS = {
    "garlic",
    "ginger",
    "ginkgo",
    "ginkgo biloba",
    "ginseng",
    "bilberry",
    "danshen",
    "dong quai",
    "feverfew",
    "evening primrose",
    "turmeric",
    "curcumin",
    "clove",
    "anise",
    "red clover",
    "salicylate",
    "willow bark",
    "piracetam",  # included in dataset examples
}

# Optionally configure Gemini
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Gemini configured with model: {GEMINI_MODEL_NAME}")
else:
    gemini_model = None
    if not GEMINI_AVAILABLE:
        print("[INFO] google-generativeai is not installed. Gemini will not be used.")
    elif not GEMINI_API_KEY:
        print("[INFO] GEMINI_API_KEY is not set. Gemini will not be used.")

# Lazy-loaded local Arabic model (only when needed)
local_ar_pipeline = None


# ==========================
# Utility: Save Arabic text output to file
# ==========================


def save_output_to_file(filename: str, text: str, output_dir: str = "outputs"):
    """
    Save Arabic explanation to UTF-8 file.
    Creates the outputs directory if missing.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVED] Arabic explanation saved to: {path}")


# ==========================
# 3. Utilities
# ==========================


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


def resolve_food_name_semantically(food_raw: str, similarity_threshold: float = 0.6):
    """
    Given a user-entered food/herb string (can be Arabic or English),
    find the most similar food/herb from the dataset using embeddings.

    Returns:
        (resolved_food, best_similarity)

    If best_similarity < similarity_threshold, we keep the original food_raw.
    """
    if food_raw is None:
        return food_raw, 0.0

    text = str(food_raw).strip()
    if not text:
        return text, 0.0

    # Embed the user text
    query_emb = st_model.encode(
        [text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Compute cosine similarity to all food embeddings
    sims = cosine_similarity(query_emb, FOOD_EMBS)[0]  # shape (n_foods,)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_food = FOOD_LIST[best_idx]

    if best_sim >= similarity_threshold:
        # Good semantic match → map to known food
        return best_food, best_sim

    # Otherwise, use original user text
    return food_raw, best_sim


SEVERITY_ORDER = ["info", "minor", "moderate", "severe"]


def severity_rank(label: str) -> int:
    """
    Map severity label to rank (higher index = more severe).
    """
    label = str(label)
    if label not in SEVERITY_ORDER:
        return 0
    return SEVERITY_ORDER.index(label)


# ==========================
# 0. Relevance Checker Agent
# ==========================


def relevance_checker(drug_name: str, food, interaction_text: str):
    """
    Check whether the query looks like a valid drug–food/herb interaction question.

    Returns:
        (is_relevant: bool, reason_code: str, arabic_message: str or None)
    """
    dn = (drug_name or "").strip()
    fn = (str(food) if food is not None else "").strip()
    it = (interaction_text or "").strip()

    dn_lower = dn.lower()
    fn_lower = fn.lower()
    it_lower = it.lower()

    # Very short / empty inputs
    if len(dn_lower) < 2 or len(fn_lower) < 2:
        msg = (
            "الاستفسار غير مكتمل.\n"
            "يُرجى كتابة اسم الدواء واسم الطعام أو العشبة بوضوح.\n"
            'مثال: "وارفارين + السبانخ" أو "أتورفاستاتين + عصير الجريب فروت".'
        )
        return False, "missing_drug_or_food", msg

    # Check for obviously non-medical context
    if any(
        bad in it_lower or bad in fn_lower or bad in dn_lower
        for bad in IRRELEVANT_KEYWORDS
    ):
        msg = (
            "يبدو أن السؤال لا يتعلّق بتداخل دوائي مع طعام أو عشبة.\n"
            "هذا النظام مخصص فقط لتداخلات الأدوية مع الأغذية والأعشاب والمكملات الغذائية."
        )
        return False, "irrelevant_domain", msg

    # Check that the drug appears in our known drug vocabulary (strict for safety)
    if dn_lower not in DRUG_VOCAB:
        msg = (
            f"اسم الدواء المدخَل ({dn}) غير موجود في قاعدة البيانات الحالية.\n"
            "يُرجى التأكد من تهجئة اسم الدواء كما يظهر في الوصفة الطبية أو النشرة الداخلية."
        )
        return False, "unknown_drug", msg

    # For food/herb: be more permissive. We no longer require exact match in FOOD_VOCAB.
    if len(fn_lower) < 2:
        msg = (
            "الاستفسار غير مكتمل.\n"
            "يُرجى كتابة اسم الطعام أو العشبة بشكل أوضح.\n"
            "مثال: grapefruit juice, garlic, السبانخ، الشاي الأخضر ..."
        )
        return False, "missing_food", msg

    # Very short description = still okay but we note it's weak
    if len(it.split()) < 3:
        # We still treat this as relevant, but minimal info
        return True, "low_context", None

    # Food can be anything; semantic resolver will try to map it later.
    return True, "ok", None


# ==========================
# 4. Agent 1: Retrieval Agent
# ==========================


def retrieval_agent(drug_name: str, food, interaction_text: str, k: int = K_NEIGHBORS):
    """
    Use the similarity index to retrieve top-k neighbors.
    Returns:
        {
          "query_text": ...,
          "neighbors": [ {row_id, drug_name, food, severity, interaction_text, distance, similarity, cluster_kmeans}, ... ]
        }
    """
    text = build_embedding_text(drug_name, food, interaction_text)
    query_emb = embed_query_text(text)

    distances, indices = nn_model.kneighbors(query_emb, n_neighbors=k)
    distances = distances[0]
    indices = indices[0]

    sims = 1.0 - distances
    sims = np.clip(sims, 0.0, 1.0)

    neighbor_rows = []
    for idx, dist, sim in zip(indices, distances, sims):
        row = df.iloc[idx]
        neighbor_rows.append(
            {
                "row_id": int(idx),
                "drug_name": row["drug_name"],
                "food": row["food"],
                "severity": str(row["severity"]),
                "interaction_text": row["interaction_text"],
                "distance": float(dist),
                "similarity": float(sim),
                "cluster_kmeans": (
                    int(row["cluster_kmeans"])
                    if "cluster_kmeans" in df.columns
                    else None
                ),
            }
        )

    return {"query_text": text, "neighbors": neighbor_rows}


# ==========================
# 5. Agent 2: Severity Aggregation Agent
# ==========================


def zero_shot_severity_from_neighbors(neighbor_rows):
    """
    Compute neighbor-based severity distribution and predicted label.
    """
    severity_weights = {}
    for nbr in neighbor_rows:
        sev = nbr["severity"]
        sim = nbr["similarity"]
        severity_weights.setdefault(sev, 0.0)
        severity_weights[sev] += float(sim)

    total_weight = sum(severity_weights.values()) or 1.0
    severity_probs = {sev: w / total_weight for sev, w in severity_weights.items()}

    pred_sev = max(severity_probs.items(), key=lambda x: x[1])[0]
    confidence = severity_probs[pred_sev]

    return pred_sev, confidence, severity_probs


def classifier_severity(drug_name: str, food, interaction_text: str):
    """
    Use MLP classifier to predict severity and probabilities.
    """
    text = build_embedding_text(drug_name, food, interaction_text)
    query_emb = embed_query_text(text)

    y_proba = clf.predict_proba(query_emb)[0]
    y_idx = int(np.argmax(y_proba))
    pred_label = label_encoder.inverse_transform([y_idx])[0]

    prob_dict = {
        label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(y_proba)
    }

    return str(pred_label), prob_dict


def severity_agent(drug_name: str, food, interaction_text: str, neighbor_rows):
    """
    Combine neighbor-based and classifier-based severity.
    Returns:
        {
          "final_severity": ...,
          "agreement": True/False,
          "neighbor_pred": ...,
          "neighbor_confidence": ...,
          "neighbor_distribution": {...},
          "classifier_pred": ...,
          "classifier_probs": {...},
          "confidence_level": "high"/"medium"/"low"
        }
    """
    # Neighbor-based
    nb_pred, nb_conf, nb_dist = zero_shot_severity_from_neighbors(neighbor_rows)

    # Classifier-based
    clf_pred, clf_probs = classifier_severity(drug_name, food, interaction_text)

    # Decide final severity
    if nb_pred == clf_pred:
        final = nb_pred
        agreement = True
    else:
        # Choose more conservative (more severe)
        if severity_rank(nb_pred) > severity_rank(clf_pred):
            final = nb_pred
        else:
            final = clf_pred
        agreement = False

    # Confidence heuristic
    if agreement and nb_conf >= 0.7:
        confidence_level = "high"
    elif agreement or nb_conf >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    return {
        "final_severity": final,
        "agreement": agreement,
        "neighbor_pred": nb_pred,
        "neighbor_confidence": nb_conf,
        "neighbor_distribution": nb_dist,
        "classifier_pred": clf_pred,
        "classifier_probs": clf_probs,
        "confidence_level": confidence_level,
    }


# ==========================
# 6. Agent 3: Clinical Reasoning Agent (English)
# ==========================


def infer_mechanism_and_risk(
    drug_name: str, food, interaction_text: str, neighbor_rows, final_severity: str
):
    """
    Rule-based mechanism inference using keywords + injectable override.
    Produces an English summary.
    """
    txt_all = (
        (interaction_text or "")
        + " "
        + " ".join(n["interaction_text"] or "" for n in neighbor_rows)
    )
    txt_all_lower = txt_all.lower()

    drug_lower = str(drug_name).lower()
    food_lower = str(food).lower()

    is_injectable = drug_lower in INJECTABLE_DRUGS

    mechanism = "Unclear or nonspecific interaction mechanism."
    risk = "Potential interaction; clinical significance is uncertain."
    type_label = "other"

    # Flags for patterns
    bleeding_pattern = any(
        k in txt_all_lower
        for k in ["bleeding", "hemorrhage", "anticoagulant", "antiplatelet"]
    ) or any(h in food_lower for h in ANTICOAGULANT_FOODS)

    grapefruit_pattern = "grapefruit" in txt_all_lower or "cyp3a4" in txt_all_lower

    vitamin_k_pattern = (
        "vitamin k" in txt_all_lower
        or "leafy" in txt_all_lower
        or "spinach" in txt_all_lower
    )

    absorption_pattern = (
        any(k in txt_all_lower for k in ["bioavailability", "absorption"])
        or "take with food" in txt_all_lower
    )

    alcohol_pattern = "alcohol" in txt_all_lower or "ethanol" in txt_all_lower

    # Injectable override logic
    if is_injectable:
        # 1) Injectable + anticoagulant herb → bleeding mechanism
        if bleeding_pattern:
            mechanism = (
                "Additive anticoagulant or antiplatelet effect between the injectable drug "
                "and the herb/food, increasing the risk of bleeding."
            )
            risk = (
                "Increased risk of bleeding, bruising, or hemorrhage, especially in high-risk patients "
                "or when combined with other anticoagulant therapies."
            )
            type_label = "bleeding"

        # 2) Injectable + alcohol → alcohol mechanism
        elif alcohol_pattern:
            mechanism = "Alcohol may potentiate some hemodynamic or central nervous system effects of the drug."
            risk = "Potential for hypotension, dizziness, or other adverse effects depending on the clinical context."
            type_label = "alcohol"

        # 3) Injectable + no strong pattern → essentially no food-based mechanism
        else:
            mechanism = (
                "This drug is administered intravenously or by injection and does not rely on "
                "gastrointestinal absorption, so typical food interactions are not expected."
            )
            risk = (
                "No well-established direct interaction with common foods or beverages; "
                "the main safety concerns relate to the drug's intrinsic effects."
            )
            type_label = "injectable_none"

    else:
        # Non-injectable (oral or unknown): use standard patterns
        if bleeding_pattern:
            mechanism = "Additive anticoagulant or antiplatelet effect, increasing the risk of bleeding."
            risk = "Increased risk of bleeding, bruising, or hemorrhage, especially in high-risk patients."
            type_label = "bleeding"

        if grapefruit_pattern:
            mechanism = "Inhibition of CYP3A4 by grapefruit or similar components, leading to increased drug blood levels."
            risk = "Higher exposure to the drug and increased risk of dose-related adverse effects or toxicity."
            type_label = "cyp3a4"

        if vitamin_k_pattern:
            mechanism = "Vitamin K from leafy green vegetables may oppose the anticoagulant effect of warfarin."
            risk = "Reduced anticoagulant effect and possible increase in clotting risk if intake is inconsistent."
            type_label = "vitamin_k"

        if absorption_pattern:
            mechanism = "Food affects drug absorption or gastrointestinal tolerability."
            risk = "Changes in drug levels or GI side effects depending on whether the drug is taken with or without food."
            if type_label == "other":
                type_label = "absorption"

        if alcohol_pattern:
            mechanism = "Alcohol interacts with the drug, potentially increasing CNS depression or liver toxicity."
            risk = "Increased risk of sedation, liver injury, or other adverse effects depending on the drug."
            type_label = "alcohol"

    # Build a plain-English severity description
    if final_severity == "severe":
        severity_english = (
            "This interaction is considered high risk and potentially serious."
        )
    elif final_severity == "moderate":
        severity_english = "This interaction is considered moderate risk and may be clinically significant."
    elif final_severity == "minor":
        severity_english = "This interaction is considered low to mild risk."
    else:
        severity_english = "This interaction is mostly informational or low risk."

    # For injectable drugs where we detected no specific mechanism,
    # emphasize that food interaction is limited/uncertain.
    if is_injectable and type_label == "injectable_none":
        severity_english = (
            "For this parenteral (injectable) drug, clinically significant food interactions are not well established. "
            "The severity classification here should be interpreted cautiously."
        )

    return {
        "mechanism": mechanism,
        "risk": risk,
        "type_label": type_label,
        "severity_english": severity_english,
    }


def clinical_reasoning_agent(
    drug_name: str, food, interaction_text: str, neighbor_rows, severity_info
):
    """
    Compose a structured English reasoning bundle including mechanism, risk,
    and recommendations based on severity.
    """
    final_sev = severity_info["final_severity"]
    mech_risk = infer_mechanism_and_risk(
        drug_name, food, interaction_text, neighbor_rows, final_sev
    )

    # Basic recommendation heuristics (severity-driven)
    if final_sev == "severe":
        recommendation = (
            "Avoid this food or supplement while taking the drug whenever possible, "
            "and consult a physician or pharmacist to discuss alternatives or monitoring."
        )
    elif final_sev == "moderate":
        recommendation = (
            "Use caution and monitor for changes in drug effect or side effects. "
            "Consult a healthcare professional for individualized advice."
        )
    elif final_sev == "minor":
        recommendation = (
            "The interaction is generally mild. Follow the usual instructions and monitor for minor changes; "
            "discuss with your healthcare provider if you have concerns."
        )
    else:  # info
        recommendation = (
            "This interaction is mostly informational. Maintain consistent dietary habits and "
            "discuss any major changes with your healthcare provider."
        )

    # Select a few neighbor examples (for LLM to justify explanation)
    example_texts = []
    for nbr in neighbor_rows[:3]:
        example_texts.append(
            f"- Drug: {nbr['drug_name']}, Food: {nbr['food']}, Severity: {nbr['severity']}. "
            f"Interaction: {nbr['interaction_text']}"
        )

    reasoning_bundle = {
        "drug_name": str(drug_name),
        "food": str(food),
        "final_severity": final_sev,
        "severity_info": severity_info,
        "mechanism": mech_risk["mechanism"],
        "risk": mech_risk["risk"],
        "severity_english": mech_risk["severity_english"],
        "recommendation": recommendation,
        "neighbor_examples": example_texts,
    }

    return reasoning_bundle


# ==========================
# 7. Agent 4: Arabic Explanation Agent
# ==========================


def init_local_arabic_pipeline():
    """
    Lazy-load local multilingual model for Arabic generation.
    """
    global local_ar_pipeline
    if local_ar_pipeline is not None:
        return local_ar_pipeline

    if not TRANSFORMERS_AVAILABLE:
        print("[WARN] transformers not installed; local Arabic model not available.")
        return None

    try:
        print(f"Loading local Arabic/multilingual model: {LOCAL_AR_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_AR_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_AR_MODEL_NAME)
        local_ar_pipeline = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer
        )
        print("Local Arabic model loaded.")
    except Exception as e:
        print(f"[WARN] Failed to load local Arabic model: {e}")
        local_ar_pipeline = None

    return local_ar_pipeline


def build_arabic_prompt(reasoning_bundle):
    """
    Build a concise English instruction for the LLM, asking it to answer in Arabic.
    """
    drug = reasoning_bundle["drug_name"]
    food = reasoning_bundle["food"]
    user_food = reasoning_bundle.get("user_entered_food", food)
    resolved_food = reasoning_bundle.get("semantically_resolved_food", food)
    final_sev = reasoning_bundle["final_severity"]
    severity_english = reasoning_bundle["severity_english"]
    mechanism = reasoning_bundle["mechanism"]
    risk = reasoning_bundle["risk"]
    recommendation = reasoning_bundle["recommendation"]
    examples = "\n".join(reasoning_bundle["neighbor_examples"])

    # For clarity to the LLM, mention both user-entered and resolved food if they differ
    if user_food != resolved_food:
        food_line = (
            f"{user_food} (approximately matched to '{resolved_food}' in the database)"
        )
    else:
        food_line = resolved_food

    prompt = f"""
You are a clinical pharmacist. Write a clear, concise explanation in **Modern Standard Arabic** for a patient about a possible drug–food or drug–herb interaction.

Keep the style simple and understandable for non-experts. Do NOT give dosing changes, and do NOT tell the patient to stop or adjust the medication dose by themselves. Always recommend consulting a doctor or pharmacist.

Information:
- Drug: {drug}
- Food/herb: {food_line}
- Final severity (English): {final_sev}  ({severity_english})
- Mechanism (English): {mechanism}
- Main risk (English): {risk}
- General recommendation (English): {recommendation}

Some similar known interactions from a trusted database:
{examples}

Please respond ONLY in Arabic, with the following structure:

1) Brief line with the drug and food.
2) Line for severity: use words like "شديدة" / "متوسطة" / "بسيطة" / "معلوماتية فقط".
3) Short paragraph explaining what the interaction means and the possible risks.
4) Short paragraph with practical advice, always including a recommendation to استشارة الطبيب أو الصيدلي.
5) A final warning that this information does not replace medical consultation.
"""
    return prompt


def arabic_explanation_agent(reasoning_bundle):
    """
    Try Gemini first. If it fails, fall back to local model.
    If that fails, use a simple hard-coded Arabic template.
    """
    prompt = build_arabic_prompt(reasoning_bundle)

    # 1) Try Gemini
    if gemini_model is not None:
        try:
            resp = gemini_model.generate_content(prompt)
            if hasattr(resp, "text") and resp.text:
                arabic_text = resp.text.strip()
            else:
                arabic_text = str(resp).strip()
            return (
                arabic_text
                + "\n\n⚠️ تنبيه: هذه المعلومات عامة ولا تُعد بديلاً عن استشارة الطبيب أو الصيدلي. لا تُوقِف أو تُغيِّر جرعة الدواء من تلقاء نفسك."
            )
        except Exception as e:
            print(f"[WARN] Gemini Arabic generation failed: {e}")

    # 2) Local Transformers model
    pipe = init_local_arabic_pipeline()
    if pipe is not None:
        try:
            out = pipe(prompt, max_length=512, truncation=True)
            text = out[0]["generated_text"].strip()
            return (
                text
                + "\n\n⚠️ تنبيه: هذه المعلومات عامة ولا تُعد بديلاً عن استشارة الطبيب أو الصيدلي. لا تُوقِف أو تُغيِّر جرعة الدواء من تلقاء نفسك."
            )
        except Exception as e:
            print(f"[WARN] Local Arabic model generation failed: {e}")

    # 3) Last-resort simple template (English mechanism but Arabic wrapper)
    drug = reasoning_bundle["drug_name"]
    food = reasoning_bundle["food"]
    final_sev = reasoning_bundle["final_severity"]

    sev_map = {
        "severe": "شديدة",
        "moderate": "متوسطة",
        "minor": "بسيطة",
        "info": "معلوماتية فقط",
    }
    sev_ar = sev_map.get(final_sev, "غير معروفة")

    mech = reasoning_bundle["mechanism"]
    risk = reasoning_bundle["risk"]
    rec = reasoning_bundle["recommendation"]

    fallback_text = f"""🔶 الدواء: {drug}
🔶 الطعام/العشبة: {food}

🔸 درجة الخطورة: {sev_ar}
قد يكون هناك تداخل بين هذا الدواء وهذا الطعام/العشبة. (وصف آلي باللغة الإنجليزية): {mech}
المخاطر المحتملة (وصف آلي باللغة الإنجليزية): {risk}

🔸 ما الذي يُنصح به؟
{rec}

⚠️ تنبيه: هذه المعلومات عامة ولا تُعد بديلاً عن استشارة الطبيب أو الصيدلي. لا تُوقِف أو تُغيِّر جرعة الدواء من تلقاء نفسك.
"""
    return fallback_text


# ==========================
# 8. Agent 5: Safety Checker
# ==========================


def safety_checker(arabic_text: str, final_severity: str, confidence_level: str) -> str:
    """
    Append extra safety notes depending on severity and confidence.
    """
    extra_lines = []

    if final_severity in ["severe", "moderate"]:
        extra_lines.append(
            "⚠️ إذا كنت تستخدم هذا الدواء حالياً، لا تُغيِّر أي شيء بدون مراجعة الطبيب أو الصيدلي. "
            "قد يتطلّب الأمر مراقبة أدق أو تغييراً في الخطة العلاجية."
        )

    if confidence_level in ["low", "medium"]:
        extra_lines.append(
            "ℹ️ ملاحظة: المعلومات عن هذا التداخل قد تكون غير كاملة، لذلك من المهم مناقشة وضعك الشخصي مع مختص صحي."
        )

    if extra_lines:
        return arabic_text.strip() + "\n\n" + "\n".join(extra_lines)
    return arabic_text


# ==========================
# 9. Full Pipeline: From Query → Arabic Explanation
# ==========================


def explain_interaction_in_arabic(
    drug_name: str, food, interaction_text: str, k: int = K_NEIGHBORS
) -> dict:
    """
    Main entry point:
        - Runs relevance checker
        - If relevant: resolves food name semantically
        - Runs all agents
        - If not: returns only a short Arabic message
    """
    is_rel, reason_code, rel_msg = relevance_checker(drug_name, food, interaction_text)

    if not is_rel:
        # Irrelevant or out-of-scope → return early
        arabic_text = (
            rel_msg + "\n\n⚠️ تنبيه: هذه الأداة لا تُغني عن استشارة الطبيب أو الصيدلي."
        )
        return {
            "drug_name": drug_name,
            "food": food,
            "interaction_text": interaction_text,
            "relevant": False,
            "relevance_reason": reason_code,
            "neighbors": [],
            "severity_info": None,
            "reasoning_bundle": None,
            "arabic_explanation": arabic_text,
        }

    # Relevant → try to resolve the food name semantically
    resolved_food, food_sim = resolve_food_name_semantically(food)

    # Agent 1: Retrieval
    retrieval = retrieval_agent(drug_name, resolved_food, interaction_text, k=k)
    neighbors = retrieval["neighbors"]

    # Agent 2: Severity
    severity_info = severity_agent(
        drug_name, resolved_food, interaction_text, neighbors
    )

    # Agent 3: Clinical Reasoning (English)
    reasoning_bundle = clinical_reasoning_agent(
        drug_name, resolved_food, interaction_text, neighbors, severity_info
    )

    # Add mapping info for the Arabic agent (and for debugging)
    reasoning_bundle["user_entered_food"] = str(food)
    reasoning_bundle["semantically_resolved_food"] = str(resolved_food)
    reasoning_bundle["food_similarity_score"] = float(food_sim)

    # Agent 4: Arabic Explanation
    arabic_core = arabic_explanation_agent(reasoning_bundle)

    # Agent 5: Safety Checker
    arabic_final = safety_checker(
        arabic_core, severity_info["final_severity"], severity_info["confidence_level"]
    )

    return {
        "drug_name": drug_name,
        "food": food,  # original user input
        "resolved_food": resolved_food,  # mapped dataset value (if any)
        "food_similarity_score": food_sim,
        "interaction_text": interaction_text,
        "relevant": True,
        "relevance_reason": "ok",
        "neighbors": neighbors,
        "severity_info": severity_info,
        "reasoning_bundle": reasoning_bundle,
        "arabic_explanation": arabic_final,
    }


# ==========================
# 10. Demo
# ==========================


def demo_case(drug_name: str, food, interaction_text: str, filename: str):
    print("\n" + "=" * 100)
    print(f"EXPLAIN INTERACTION (ARABIC) DEMO")
    print("=" * 100)
    print(f"Drug: {drug_name}")
    print(f"Food: {food}")
    print(f"Interaction (user text): {interaction_text}")
    print("-" * 100)

    result = explain_interaction_in_arabic(
        drug_name, food, interaction_text, k=K_NEIGHBORS
    )

    if not result["relevant"]:
        print("\n[Relevance checker] Query deemed NOT relevant:")
        print("  Reason:", result["relevance_reason"])
        print("Arabic message saved to file.")
        save_output_to_file(filename, result["arabic_explanation"])
        print("=" * 100 + "\n")
        return

    # Print severity info in terminal (English = readable)
    print("\n[Final severity info]")
    print("  Final severity:", result["severity_info"]["final_severity"])
    print("  Confidence level:", result["severity_info"]["confidence_level"])
    print("  Resolved food:", result.get("resolved_food"))
    print("  Food similarity score:", result.get("food_similarity_score"))

    print("\nArabic explanation saved to file (not printed here).")

    # Save Arabic explanation
    save_output_to_file(filename, result["arabic_explanation"])

    print("=" * 100 + "\n")


if __name__ == "__main__":
    # Example 1: Atorvastatin + grapefruit juice
    demo_case(
        drug_name="Atorvastatin",
        food="grapefruit juice",
        interaction_text="I am taking atorvastatin. Is it safe to drink grapefruit juice?",
        filename="atorvastatin_grapefruit.txt",
    )

    # Example 2: Fluoxetine + aged cheese (your custom example)
    demo_case(
        drug_name="Fluoxetine",
        food="aged cheese",
        interaction_text="I'm taking Fluoxetine with aged cheese, is that ok?",
        filename="Fluoxetine.txt",
    )

    # Example 3: Warfarin + leafy green vegetables
    demo_case(
        drug_name="Warfarin",
        food="leafy green vegetables",
        interaction_text="I am on warfarin and I eat a lot of leafy green vegetables.",
        filename="warfarin_leafy_greens.txt",
    )

    # Example 4: Metformin + food
    demo_case(
        drug_name="Metformin",
        food="food",
        interaction_text="My doctor told me to take metformin with food. Why is that?",
        filename="metformin_food.txt",
    )

    # Example 5: Clearly irrelevant example to test relevance checker
    demo_case(
        drug_name="Windows 10",
        food="banana",
        interaction_text="Is Windows 10 compatible with banana?",
        filename="irrelevant_example.txt",
    )

    print(
        "Task 8: multi-agent Arabic explanations demo completed. Files saved in /outputs."
    )
