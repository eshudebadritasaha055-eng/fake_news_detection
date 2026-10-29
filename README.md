# Explainable Fake News Detection Using BERT with Comparative XAI (LIME and Anchors)
### CA2 — Feature Extraction, Model Building & Final Project Submission (Phase 2)

**Course:** Natural Language Processing (`25MCAA1303`)  
**Programme:** MCA-AIML (Semester III, 2026–2027)  
**Institution:** School of Computer Science and IT, Jain (Deemed-to-be University)  
**Faculty In-Charge:** Dr. Sharal Coelho, Assistant Professor, Dept. of CS & IT  

---

## Team Members & Responsibilities

| Team Member | Registration No. | Contributions Across Phase 1 and Phase 2 |
|---|---|---|
| **Rithish Natarajan E** | `25MCAR0016` | System architecture design; Module 1 (`bert-base-uncased` fine-tuning, hyperparameter tuning, and TF-IDF baseline benchmarking); GPU inference wrapper; joint end-to-end error analysis and report compilation. |
| **Debadrita Saha** | `25MCAR0028` | Dataset curation and validation (LIAR partitions and external headline sample); bifurcated preprocessing pipeline (negation-preserving cleaning vs. classical lemmatization); feature extraction; joint evaluation and report compilation. |
| **Ezhilarasan M** | `25MCAR0030` | Literature synthesis; Module 2 (LIME token attribution) and Module 3 (Anchors rule extraction & comparative Jaccard overlap audit); Streamlit dashboard integration; joint error diagnosis and report compilation. |

---

## Project Overview

Neural text classifiers such as BERT achieve state-of-the-art performance on short-statement misinformation detection, but operate as opaque black boxes. In high-stakes misinformation auditing, an unexplained probability cannot be trusted: neither an end user nor an engineer can tell whether the classifier relied on meaningful linguistic deception patterns or memorized spurious shortcuts (e.g., sensationalist framing).

This project implements and evaluates an integrated, end-to-end Natural Language Processing and Explainable AI pipeline that:
1. **Classifies** short statements and news headlines as **Fake** or **Real** using both classical machine learning baselines (Multinomial Naive Bayes, Logistic Regression, Linear SVM) and a fine-tuned `bert-base-uncased` transformer.
2. **Attributes** token-level continuous directionality via **LIME** (Local Interpretable Model-Agnostic Explanations).
3. **Extracts** high-precision, sufficient IF-THEN rules via **Anchors** ($\tau \ge 0.95$).
4. **Audits** agreement, stability, and runtime trade-offs between both explainability paradigms across high-confidence, borderline, and cross-domain news headline regimes.

```
+---------------------------------------------------------------------------------------------------------+
|                                        Integrated System Pipeline                                       |
+---------------------------------------------------------------------------------------------------------+
|  [Input Claim]                                                                                          |
|        │                                                                                                |
|        ├──► Bifurcated Preprocessing:                                                                   |
|        │       ├─► Classical ML: Lowercasing, Punctuation Stripping, Lemmatization, Stopword Removal    |
|        │       └─► BERT Contextual: Light Cleaning, Punctuation & Negation Preservation                 |
|        │                                                                                                |
|  [Module 1: Classification]                                                                             |
|        ├─► Baselines: Multinomial NB | Logistic Regression | Linear SVM                                 |
|        └─► Proposed: Fine-Tuned BERT (bert-base-uncased) ──► P(Fake) & P(Real)                          |
|                                                                                                         |
|  [Comparative Post-Hoc XAI Layer (Stateless Batched Wrapper)]                                           |
|        ├─► Module 2 (LIME): N=1,000 Perturbations ──► Continuous Token Attribution Weights            |
|        └─► Module 3 (Anchors): Beam Search (τ ≥ 0.95) ──► Minimal IF-THEN Sufficiency Decision Rule    |
|                                                                                                         |
|  [Comparative Audit & Diagnostics]                                                                      |
|        ├─► Top-3 Token Overlap & Jaccard Agreement Audit                                                |
|        ├─► Sensationalist Lexical Bias Detection (Debugging High-Confidence False Positives)            |
|        └─► Borderline Divergence Warning (Surfacing Decision Boundary Instability)                      |
+---------------------------------------------------------------------------------------------------------+
```

---

## Repository Structure

```
Debadrita/
├── Explainable Fake News Detection Using BERT with Comparative XAI — Final Report.pdf  # Final written report (12 pages)
├── Explainable Fake News Detection Using BERT with Comparative XAI (2).pptx           # Final presentation slide deck
├── README.md                                                                           # GitHub documentation
├── requirements.txt                                                                    # Environment dependencies
├── app.py                                                                              # Interactive Streamlit dashboard
├── notebooks/                                                                          # 4 sequential Jupyter notebooks
│   ├── 01_eda_and_preprocessing.ipynb                                                 # Data inspection & bifurcated cleaning
│   ├── 02_module1_baselines_and_bert.ipynb                                             # Classical models & BERT fine-tuning
│   ├── 03_modules2_3_comparative_xai.ipynb                                             # LIME & Anchors comparative audit
│   ├── 04_error_analysis_and_cross_domain.ipynb                                        # Error diagnosis & cross-domain shift
│   ├── data/                                                                           # Raw dataset partitions
│   │   ├── train.tsv                                                                   # LIAR training partition (10,240 claims)
│   │   ├── valid.tsv                                                                   # LIAR validation partition (1,284 claims)
│   │   ├── test.tsv                                                                    # LIAR test partition (1,267 claims)
│   │   └── secondary_validation_headlines.csv                                          # ISOT/FakeNewsNet headlines (300 items)
│   └── processed/                                                                      # Clean bifurcated splits
│       ├── liar_clean_train.csv
│       ├── liar_clean_valid.csv
│       ├── liar_clean_test.csv
│       └── external_headlines_clean.csv
└── models/                                                                             # Trained models, vectorizers & reports
    ├── tfidf_vectorizer.joblib                                                         # 5,000-feature unigram/bigram TF-IDF
    ├── multinomial_nb.joblib                                                           # Trained Multinomial Naive Bayes
    ├── logistic_regression.joblib                                                      # Trained Logistic Regression
    ├── linear_svm.joblib                                                               # Trained Calibrated Linear SVM
    ├── bert_classifier_metadata.json                                                   # BERT config & training hyperparameters
    ├── module1_classification_report.csv                                               # Test performance across models (Table 1)
    ├── module1_confusion_matrices.json                                                 # Test confusion matrices (Figure 2)
    ├── module2_lime_audit_summary.csv                                                  # LIME attribution audit summary
    ├── module3_anchors_audit_summary.csv                                               # Anchors rule extraction summary
    ├── comparative_xai_audit_results.csv                                               # Confidence-stratified audit (Table 2)
    └── cross_domain_evaluation.csv                                                     # Cross-domain benchmark (Table 3)
```

---

## Datasets & Preprocessing Strategy

1. **LIAR Benchmark Dataset (Primary Corpus)**:
   - Curated by Wang (2017) from PolitiFact's fact-checking archive.
   - Total of 12,836 statements (train: 10,240; valid: 1,284; test: 1,267) averaging 17.9 words.
   - **Binary Mapping**:
     - **Fake (Class 0)**: `pants-fire`, `false`, `barely-true` (44.1%)
     - **Real (Class 1)**: `half-true`, `mostly-true`, `true` (55.9%)
   - **Metadata Exclusion**: Speaker credit-history counts (columns 8–12) are strictly excluded. Feeding speaker counts artificially inflates accuracy above 72% but degrades the model to a lookup table rather than an NLP text classifier.
2. **Secondary Validation Pool (ISOT / FakeNewsNet Headlines)**:
   - 300 held-out headlines (150 Fake, 150 Real) averaging 12.4 words to evaluate zero-shot domain transfer from political speech to breaking news.
3. **Bifurcated Preprocessing**:
   - *Classical ML*: Strips stopwords, punctuation, and applies WordNet lemmatization.
   - *BERT Contextual*: Light cleaning only. Strictly preserves punctuation, capitalization, and semantic negation words (`not`, `no`, `never`, `without`, `nor`). Stripping negation words causes truth-condition inversion and breaks Anchors rule stability.

---

## Reproducing the Results

### 1. Environment Installation
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

### 2. Running the Notebooks
Execute the notebooks in numerical sequence:
- `01_eda_and_preprocessing.ipynb`: Inspects data distributions, validates binary balance, and exports clean CSVs.
- `02_module1_baselines_and_bert.ipynb`: Fits TF-IDF representations, trains MNB, LR, SVM, and configures BERT fine-tuning.
- `03_modules2_3_comparative_xai.ipynb`: Runs the stateless batched wrapper, extracts LIME token weights and Anchors rules, and conducts the 100-sample comparative audit.
- `04_error_analysis_and_cross_domain.ipynb`: Deep-dives into sensational false positives, borderline explanation divergence, and zero-shot cross-domain testing.

### 3. Launching the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
The interface provides interactive claim input, model selection, live LIME bar charts, Anchors rule display, Jaccard overlap metrics, and automated audit diagnostic flags.

---

## Note on the `models/` Directory

All classical ML models (`linear_svm.joblib`, `logistic_regression.joblib`, `multinomial_nb.joblib`), vectorizers, and audit metric CSVs are committed directly. 
**Pretrained BERT transformer weights (~440MB) exceed GitHub's 100MB per-file upload limit**: full transformer architecture metadata and reproducible training scripts are defined in `models/bert_classifier_metadata.json` and `notebooks/02_module1_baselines_and_bert.ipynb`.

---

## Key Empirical Results

### Table 1: Module 1 Classification Performance on LIAR Test Split ($n = 1,283$)

| Model | Accuracy | Precision (Fake) | Recall (Fake) | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Multinomial Naive Bayes (TF-IDF)** | 0.596 | 0.548 | 0.465 | 0.578 | 0.624 |
| **Logistic Regression (TF-IDF, balanced)** | 0.605 | 0.551 | 0.546 | 0.598 | 0.641 |
| **Linear SVM (TF-IDF, calibrated)** | 0.612 | 0.562 | 0.530 | 0.602 | 0.648 |
| **Fine-Tuned BERT (`bert-base-uncased`)** | **0.648** | **0.597** | **0.611** | **0.642** | **0.694** |

*Fine-tuned BERT achieves the highest performance across all metrics, increasing Fake recall from 0.465 to 0.611 (detecting 338 deceptive claims vs. 257 for Naive Bayes).*

---

### Table 2: Comparative Evaluation of LIME and Anchors Across Confidence Regimes

| Prediction Regime (LIAR Test Audit, $n = 100$) | LIME Time (s) | Anchors Time (s) | Mean Anchor Length | Mean Rule Coverage | Top-3 Token Agreement |
|---|---|---|---|---|---|
| **High-Confidence Correct** ($P \ge 0.80, n = 35$) | 1.39 s | 3.92 s | 2.04 words | 0.294 (29.4%) | **85.7% (Jaccard 0.71)** |
| **High-Confidence Error** ($P \ge 0.80, n = 15$) | 1.45 s | 4.18 s | 2.20 words | 0.268 (26.8%) | **73.3% (Jaccard 0.61)** |
| **Borderline Predictions** ($0.50 \le P < 0.60, n = 50$) | 1.43 s | 5.74 s | 4.62 words | 0.064 (6.4%) | **38.0% (Jaccard 0.24)** |
| **Overall Audit Subset Average** ($n = 100$) | **1.42 s** | **4.85 s** | **3.35 words** | **0.175 (17.5%)** | **60.0% (Jaccard 0.46)** |

---

### Table 3: In-Domain (LIAR) vs. Zero-Shot Cross-Domain (ISOT/FakeNewsNet Headlines)

| Evaluation Corpus | Domain Register | Linear SVM Acc. | BERT Accuracy | BERT Macro F1 | XAI Agreement |
|---|---|---|---|---|---|
| **LIAR Test Split** ($n = 1,283$) | In-domain political claims | 0.612 | 0.648 | 0.642 | 60.0% overall |
| **External Headlines** ($n = 300$) | Cross-domain news headlines | 0.533 | 0.587 | 0.579 | 51.3% overall |

---

## Diagnostic Findings & Failure Modes

1. **Sensationalist Lexical Bias (High-Confidence False Positives)**:
   - Auditing the 236 False Positives in BERT's confusion matrix revealed that factually true statements containing dramatic intensifiers (e.g., *"The state budget deficit secretly ballooned by nearly every metric over the decade"*) trigger **86.4% Fake predictions**.
   - Both LIME (`secretly`: +0.38, `every`: +0.19) and Anchors (`IF "secretly" AND "ballooned" -> FAKE`, precision 93.8%) confirm that BERT latched onto stylistic framing rather than factual implausibility.
2. **Borderline Explanation Divergence**:
   - When classifier confidence falls between 50% and 60%, top-3 token agreement drops to 38.0% and rule coverage drops to 6.4%. 
   - Disagreement between LIME and Anchors serves as an automated **Uncertainty Alarm** alerting users not to trust predictions on unstable decision boundaries.

---

## AI Usage Acknowledgement

In accordance with course evaluation policy, the team acknowledges that generative AI tools were used strictly for grammatical refinement, report formatting, and debugging syntax for the batched PyTorch-to-LIME wrapper function. All dataset preprocessing, model training, metric computation, comparative XAI experiments, and error analyses were executed and verified directly by the team members.
