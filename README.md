# Malicious JavaScript Detection using Graph Embeddings and Synthetic Data Augmentation

This repository contains the full implementation of the MSc thesis:

**“Combining Graph Node Embedding and Random Forest for Malicious JavaScript Detection”**  
*University of Genoa – MSc in Computer Engineering (Software Platform & Cybersecurity)*

Author: **Mohammad Erfan Shahabi**  
Supervisors: **Prof. Alessandro Carrega**, **Prof. Massimo Maresca**

---

## Project Overview

Modern JavaScript-based malware leverages obfuscation, code mutation, and dynamic structure to evade traditional signature-based defenses.  
This thesis proposes two complementary detection pipelines:

### Pipeline A — Graph-Based Detection  
Transforms each JavaScript sample into a k-NN similarity graph, then uses Node2Vec to generate structural embeddings that capture the relationships among samples.

### Pipeline B — Feature-Based Detection with CTGAN Augmentation  
Trains classical ML models on engineered numerical features, enhanced with synthetic benign samples generated via CTGAN to improve class balance and generalization.

Both pipelines are evaluated across a comprehensive suite of classifiers and metrics.

---

## Repository Structure

```text
malicious-js-detection/
│
├── src/
│   ├── model_with_graph_embedding.py        # Pipeline A (Graph + Node2Vec)
│   └── model_without_graph_embedding.py     # Pipeline B (CTGAN + Features)
│
├── appendix/
│   ├── pseudocode.md        # Full pseudocode for both pipelines
│   ├── configuration.md     # Complete reproduction environment
│   └── code_samples.md      # Key code excerpts used in the thesis
│
├── results/
│   └── README.md            # Place for ROC curves, tables, plots
│
├── requirements.txt         # All dependency libraries
└── README.md                # You are here
```

---

## Methodology Summary

### Pipeline A — Graph + Node2Vec
1. Robust scaling of all numerical features  
2. k-NN graph construction (k=10)  
3. Graph conversion using NetworkX  
4. Node2Vec embedding (64 dimensions, walk_length=30, num_walks=5)  
5. ML training using:
   - Random Forest  
   - LightGBM  
   - XGBoost  
   - CatBoost  
   - SVM  
   - KNN  
   - LDA / QDA  
   - MLP, Logistic Regression, etc.  
6. Cross-validation over 5 folds  
7. ROC curve generation and full metric suite  

### Pipeline B — Feature-Based + CTGAN
1. Train–test split via StratifiedKFold  
2. CTGAN loads pretrained generator: `models/js_train.pkl`  
3. Sample additional benign synthetic samples  
4. Combine synthetic and real samples into an augmented training set  
5. Train the same classifier set as Pipeline A  
6. Evaluate on the original test split  
7. Compare robustness, AUROC, and generalization between pipelines  

---

## Metrics Reported

Both pipelines evaluate:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUROC  
- Specificity  
- Matthews Correlation Coefficient (MCC)  
- Cohen’s Kappa  
- Macro & Weighted metrics  
- Geometric Mean (G-Mean)  

All results are averaged across the 5 folds.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline A (Graph-based)

```bash
python src/model_with_graph_embedding.py
```

### 3. Run Pipeline B (CTGAN-based)

```bash
python src/model_without_graph_embedding.py
```

---

## Reproducibility

All experimental reproduction details—including:

- dataset preprocessing  
- encoding and scaling  
- graph configuration and Node2Vec parameters  
- CTGAN setup and sampling  
- classifier definitions and evaluation metrics  

are documented in:

- `appendix/configuration.md`  
- `appendix/pseudocode.md`  
- `appendix/code_samples.md`  

---

## Citation

If you use this repository, please cite:

Shahabi, M. E. (2025). *Combining Graph Node Embedding and Random Forest for Malicious JavaScript Detection*. MSc Thesis, University of Genoa.

---

## Highlights

- End-to-end malware detection framework  
- Integration of Node2Vec, CTGAN, and a wide range of ML models  
- Strong performance across AUROC and F1 metrics  
- Fully reproducible and transparently documented  
- Research-grade structure following academic standards
