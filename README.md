# StyleGuard 🛡️
### Explainable AI-Generated Academic Text Detection

> Detects AI-written academic abstracts using a three-layer pipeline: stylometric features, fine-tuned RoBERTa, and an explainable XGBoost ensemble with SHAP — achieving **99.5% accuracy** with per-prediction explanations.

**INFO 5731 — Computational Methods for Information Systems**  
University of North Texas, Spring 2026 | Group 2

---

## Results

| Model | Accuracy | F1 | FPR |
|---|---|---|---|
| **Ensemble (XGBoost: stylometry + RoBERTa)** | **99.50%** | **0.9950** | **0.80%** |
| Random Forest (stylometry only) | 98.40% | 0.9840 | 1.40% |
| Logistic Regression | 98.01% | 0.9801 | — |
| Linear SVM | 97.91% | 0.9791 | — |
| RF (top-3 features ablation) | 96.52% | 0.9652 | — |
| Length-only ablation | 58.60% | 0.6072 | — |

**Out-of-distribution test** (10 unseen abstracts): 80% accuracy, 100% AI recall, 2 false positives.

---

## What StyleGuard Does

Current AI detectors (GPTZero, Turnitin) give a score with no explanation. StyleGuard shows *why* a text was flagged — which specific writing patterns triggered the decision.

**Three-layer pipeline:**

```
Raw text
   │
   ├── Layer 1: 8 stylometric features → Classical ML baselines (LR, SVM, Random Forest)
   │
   ├── Layer 2: Fine-tuned RoBERTa → P(AI) probability score
   │
   └── Layer 3: XGBoost ensemble (8 features + RoBERTa score) → Final prediction + SHAP explanation
```

**Key finding:** AI-generated academic abstracts consistently differ from human writing in:
- Higher average word length (7.1–7.8 vs 5.4–6.2)
- Lower stopword ratio (0.21–0.26 vs 0.31–0.42) — denser, more formal prose
- Longer sentences (25–29 words vs 19–22)
- More uniform sentence length (low variance)

---

## Project Structure

```
StyleGuard/
├── 01_collect_human_data.py          # Collect pre-2021 arXiv abstracts
├── 02_prepare_prompt_pack.py         # Build prompts for AI generation
├── 03_openai_direct_generate_ai.py   # Generate AI abstracts via OpenAI API
├── 04_build_final_dataset.py         # Clean, deduplicate, build final CSV
├── 05_train_baselines_and_export_results.py  # Train classical ML models
├── 06_finetune_roberta.py            # Fine-tune RoBERTa (run on Kaggle GPU)
├── 07_ensemble_and_shap.py           # XGBoost ensemble + SHAP explainability
├── predict.py                        # Predict any new text (demo)
├── config.py                         # Paths and constants
├── utils.py                          # Stylometric feature extraction
├── requirements.txt
├── test_abstracts.json               # 10 sample abstracts for testing
└── data/
    ├── processed/
    │   ├── final_binary_dataset.csv  # 4,998 abstracts (human + AI)
    │   └── stylometric_features.csv # Precomputed features
    └── outputs/
        ├── table_model_comparison_full.csv/png
        ├── fig_ensemble_confusion_matrix.png
        ├── roberta_standalone_metrics.json
        └── shap/
            ├── fig_shap_summary_beeswarm.png
            ├── fig_shap_bar_global.png
            └── fig_shap_waterfall_*.png
```

---

## Setup & Run

### Requirements
- Python 3.11
- Mac/Linux (Windows untested)
- Kaggle account (free) for Script 06 — GPU required for RoBERTa fine-tuning

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers accelerate xgboost shap
```

### Run Order

```bash
# Step 1-4: Build dataset (requires OpenAI API key in .env)
python3.11 01_collect_human_data.py
python3.11 02_prepare_prompt_pack.py
python3.11 03_openai_direct_generate_ai.py
python3.11 04_build_final_dataset.py

# Step 5: Train classical baselines
python3.11 05_train_baselines_and_export_results.py

# Step 6: Fine-tune RoBERTa (run on Kaggle GPU — free)
# Upload 06_finetune_roberta.py + data files to Kaggle
# Download: roberta_proba_meta.csv, roberta_proba_test.csv, roberta_standalone_metrics.json
# Place them in data/outputs/

# Step 7: Ensemble + SHAP (run locally)
python3.11 07_ensemble_and_shap.py
```

### Quick Test (no setup needed if data exists)

```bash
# Predict on 10 sample abstracts
python3.11 predict.py --file test_abstracts.json

# Predict a single text
python3.11 predict.py --text "Your abstract here..."
```

---

## Dataset

- **Human baseline:** 2,499 pre-2021 arXiv abstracts (cs.AI, cs.CL, physics.optics, q-bio.GN)
- **AI samples:** 2,499 abstracts generated via GPT-4o mini with matched prompts
- **Split:** 80/20 train/test using GroupShuffleSplit (no pair leakage)
- **Total:** 4,998 abstracts

---

## Explainability

StyleGuard uses SHAP (SHapley Additive exPlanations) to explain every prediction. For each text, it shows which features pushed the model toward AI or human, and by how much.

Output plots in `data/outputs/shap/`:
- **Beeswarm plot** — global feature impact across all test samples
- **Bar chart** — mean |SHAP| feature importance
- **Waterfall plots** — per-prediction explanations for correct predictions, false positives, and false negatives

---

## Limitations

- Trained on a single AI generator (GPT-4o mini) — may not generalize to Claude, Gemini, Llama
- Narrow domain (arXiv abstracts) — performance expected to drop on other academic writing styles
- External robustness test not completed (dropped from scope due to time)
- 99.5% accuracy is likely inflated relative to real-world diverse-generator settings

---

## Team

| Name | Role |
|---|---|
| Tushar Shrivastava | DL Engineering, RoBERTa, Ensemble, SHAP |
| Shreeyash Baranwal | DL Engineering, RoBERTa |
| Jeevan Chandaka | Data Engineering, arXiv collection |
| Nishal Tej Reddy Chennu | Data Engineering, Stylometric features |
| Jimit Mineshkumar Dave | Data Engineering, Evaluation |
| Rakesh Babu Inturi | Data Engineering, Cleaning pipeline |
| Nandini Pallapu | Data Engineering, AI generation |

---

## References

1. Wu et al. (2025). A Survey on LLM-Generated Text Detection. *Computational Linguistics*, 51(1).
2. Liu et al. (2023). On the Detectability of ChatGPT Content. *arXiv:2306.05524*.
3. Li et al. (2024). MAGE: Machine-generated Text Detection in the Wild. *ACL 2024*.
4. Wang et al. (2024). SemEval-2024 Task 8: Multidomain Machine-Generated Text Detection.
