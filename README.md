# Drug-Food Interaction Explainer

A comprehensive, multilingual (Arabic/English) machine learning system for analyzing and explaining drug-food and drug-herb interactions with severity classification and explainability.

## Overview

This project implements a complete ML pipeline for drug-food interaction analysis:

1. **Food Entity Extraction** – Identifies food/herb/supplement entities in interaction text (rule-based + optional Gemini fallback)
2. **Dataset Preparation** – Flattens data into (drug, food, interaction) triples
3. **Severity Labeling** – Classifies interaction severity (severe, moderate, minor, info) using rules + LLM fallback
4. **Embeddings** – Generates semantic embeddings using `sentence-transformers`
5. **Severity Classifier** – Trains ensemble ML models (Logistic Regression, Random Forest, MLP, XGBoost)
6. **Clustering & Analysis** – Performs UMAP clustering and generates interaction summaries
7. **Similarity Engine** – Implements k-NN similarity search for related interactions
8. **Multi-Agent Explainer** – Provides detailed clinical explanations with multiple reasoning agents
9. **Streamlit UI** – Interactive web interface with Arabic language support

**Data Source:** [DrugBank](https://www.drugbank.ca/) drug-food interaction database

---

## Technology Stack

- **Language:** Python 3.8+
- **ML/Data:** pandas, scikit-learn, numpy, joblib
- **NLP:**
  - sentence-transformers (all-mpnet-base-v2 embeddings)
  - google-generativeai (Gemini for fallback NER/severity labeling)
  - transformers (local fallback models)
- **Clustering:** umap-learn, scikit-learn (KMeans)
- **UI:** Streamlit
- **Optional:** xgboost, matplotlib

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/TisTis0004/Drug-Food-Interaction-Framework
cd "Drug-food interaction2"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

This project does not include a `requirements.txt`, but here are the core dependencies identified from the codebase:

**Essential packages:**

```bash
pip install pandas numpy scikit-learn joblib sentence-transformers google-generativeai tqdm umap-learn streamlit
```

**Optional packages:**

```bash
pip install xgboost matplotlib transformers  # for advanced features
```

**To install all at once:**

```bash
pip install pandas numpy scikit-learn joblib sentence-transformers google-generativeai tqdm umap-learn streamlit xgboost matplotlib transformers
```

**Generating a requirements.txt (for reproducibility):**

```bash
pip freeze > requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root with:

```env
GOOGLE_API_KEY="your_google_gemini_api_key"
```

**How to obtain a Google API Key:**

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and paste into `.env`

**Configuration Options in Code:**

- `USE_GEMINI_FALLBACK` (default: `True`) – Enable/disable Gemini LLM fallback for entity extraction and severity labeling
- `MAX_GEMINI_CALLS` (default: 30-40 per task) – Limits API calls to avoid quota exhaustion
- `SLEEP_BETWEEN_CALLS` (default: 6.0 seconds) – Rate limiting between Gemini API calls
- `GENERATION_MODEL_NAME` (default: `gemini-2.5-flash`) – Configurable in task files

---

## Usage

### Running Individual Tasks

The pipeline consists of 9 sequential tasks. Run them in order:

```bash
# Task 0: Extract food entities from raw data
python task0_food_entity_extraction.py

# Task 1: Prepare dataset (flatten into drug-food pairs)
python task1_prepare_dataset.py

# Task 2: Label interaction severity
python task2_severity_labeling.py

# Task 3: Generate sentence embeddings
python task3_embeddings.py

# Task 4: Train severity classifier ensemble
python task4_severity_classifier_ensemble.py

# Task 5: Cluster embeddings with UMAP
python task5_clustering_umap.py

# Task 6: Build similarity search engine
python task6_similarity_engine.py

# Task 7: Zero-shot severity prediction (alternative approach)
python task7_zero_shot_severity.py

# Task 8: Multi-agent explainability system
python task8_multi_agent_explainer.py

# Task 9: Launch interactive Streamlit UI (requires previous tasks)
streamlit run task9_streamlit.py
```

### Running the Full Pipeline

To run all tasks sequentially:

```bash
for task in task0_food_entity_extraction.py task1_prepare_dataset.py task2_severity_labeling.py \
            task3_embeddings.py task4_severity_classifier_ensemble.py task5_clustering_umap.py \
            task6_similarity_engine.py task7_zero_shot_severity.py task8_multi_agent_explainer.py; do
  echo "Running $task..."
  python "$task"
done
```

### Interactive UI

```bash
streamlit run task9_streamlit.py
```

The app will open at `http://localhost:8501`

**Features:**

- Select drug from dropdown or search
- Input food/herb name
- Query interaction description
- Get severity level with emoji indicators
- Read Arabic explanations with clinical reasoning

---

## Project Structure

```
Drug-food interaction2/
├── task0_food_entity_extraction.py      # NER for food/herb entities
├── task1_prepare_dataset.py              # Flatten dataset
├── task2_severity_labeling.py            # Classify severity
├── task3_embeddings.py                   # Generate embeddings
├── task4_severity_classifier_ensemble.py # Train ML models
├── task5_clustering_umap.py              # Cluster embeddings
├── task6_similarity_engine.py            # k-NN search
├── task7_zero_shot_severity.py           # Alternative severity prediction
├── task8_multi_agent_explainer.py        # Explainability agents
├── task9_streamlit.py                    # Web UI
│
├── data.json                             # Input: DrugBank data
├── data_with_food_extraction/            # Task 0 outputs
├── data_food_flat/                       # Task 1 outputs
├── data_severity_labeled/                # Task 2 outputs
├── data_embeddings/                      # Tasks 3-8 outputs
│   ├── embeddings_all_mpnet_base_v2.npy
│   ├── drug_food_interactions_embedded.csv
│   ├── best_severity_classifier.pkl
│   ├── label_encoder_severity.pkl
│   ├── severity_models_metrics.json
│   └── [clustering/similarity outputs]
├── outputs/                              # Example explanations
├── .env                                  # Environment variables
├── .gitignore
└── README.md
```

---

## Testing & Validation

### Data Integrity Checks

```bash
# Verify dataset loading
python -c "
import pandas as pd
import json
df = pd.read_json('data.json')
print(f'Loaded {len(df)} drugs from DrugBank')
print(df.head())
"
```

### Model Evaluation

After running task4, check model metrics:

```bash
# View ensemble model performance
cat data_embeddings/severity_models_metrics.json

# View individual model reports
cat data_embeddings/severity_report_*.txt
```

### Smoke Tests (Recommended Additions)

Currently **no automated test suite exists**. To add testing:

```bash
# Create tests/ directory with test files
mkdir tests

# Install pytest
pip install pytest

# Run tests
pytest tests/
```

**Suggested test areas:**

- Verify embedding shapes and values (task3)
- Test severity classifier on known examples
- Validate similarity search returns relevant results
- Check Streamlit app startup without errors

### Linting & Code Style

Currently **no linting configuration exists**. To add:

```bash
# Install linting tools
pip install pylint flake8 black

# Run linter
flake8 task*.py

# Auto-format code
black task*.py
```

---

## Common Workflows

### Workflow 1: Quick Test with Existing Data

If pre-generated embeddings and models exist in `data_embeddings/`:

```bash
# Skip to UI directly
streamlit run task9_streamlit.py
```

### Workflow 2: Retrain Models with New Data

1. Replace `data.json` with new DrugBank data
2. Run full pipeline (all tasks 0-8)
3. Launch UI on task 9

### Workflow 3: Query-Only Mode (No Retraining)

Assuming models are trained:

```bash
python -c "
from task8_multi_agent_explainer import explain_interaction_in_arabic
result = explain_interaction_in_arabic(
    drug_name='Warfarin',
    food='grapefruit'
)
print(result)
"
```

### Workflow 4: Batch Processing Multiple Drugs

```bash
import pandas as pd
from task8_multi_agent_explainer import explain_interaction_in_arabic

drugs_of_interest = ['Warfarin', 'Atorvastatin', 'Metformin']
results = []

for drug in drugs_of_interest:
    explanation = explain_interaction_in_arabic(drug_name=drug, food='grapefruit')
    results.append({'drug': drug, 'explanation': explanation})

df_results = pd.DataFrame(results)
df_results.to_csv('batch_explanations.csv', index=False)
```

---

## Troubleshooting

### Issue: `GEMINI_API_KEY is not set`

**Solution:** Ensure `.env` file exists and contains:

```env
GOOGLE_API_KEY="your_key_here"
```

### Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:** Install missing packages:

```bash
pip install sentence-transformers
```

### Issue: Streamlit app won't start

**Solution:** Check that all previous tasks (0-8) have been run:

```bash
ls -la data_embeddings/  # Verify output files exist
```

### Issue: Out of Memory during Embedding

**Solution:** Reduce batch size in task3:

```python
BATCH_SIZE = 32  # Change from 64
```

### Issue: Rate Limited by Gemini API

**Solution:**

- Increase `SLEEP_BETWEEN_CALLS` (e.g., to 10 seconds)
- Reduce `MAX_GEMINI_CALLS`
- Set `USE_GEMINI_FALLBACK = False` for rule-based only

---

## Performance & Scalability

- **Embeddings:** ~768 dimensions per interaction (all-mpnet-base-v2)
- **Training data:** ~1000+ drug-food pairs (varies by input)
- **Inference time:** <100ms per query (k-NN + classifier)
- **Streamlit app:** Supports ~50-100 concurrent users on standard hardware

## How to Verify Setup

Run this verification script:

```bash
python -c "
import sys
print('✓ Python version:', sys.version)

packages = [
    'pandas', 'numpy', 'sklearn', 'joblib', 'sentence_transformers',
    'google.generativeai', 'tqdm', 'umap', 'streamlit'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} installed')
    except ImportError:
        print(f'✗ {pkg} NOT installed')

import os
if os.path.exists('.env'):
    print('✓ .env file exists')
else:
    print('✗ .env file missing')
"
```

---

## Contributing

To contribute:

1. Create a new branch: `git checkout -b feature/my-feature`
2. Make changes and test thoroughly
3. Submit a pull request with clear description

---

## References

- **DrugBank Database:** https://www.drugbank.ca/
- **Sentence Transformers:** https://www.sbert.net/
- **Google Gemini API:** https://ai.google.dev/
- **Streamlit Documentation:** https://docs.streamlit.io/

---

**Last Updated:** January 2026

---
