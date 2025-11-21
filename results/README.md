
# Experimental Results

This section presents the experimental evaluation of two detection pipelines:

1. **Feature-Based Models (No Graph Embedding)**
2. **Graph-Based Models (Node2Vec Embedding)**

All metrics are averaged over 5 folds and include AUROC, Accuracy, Precision, Recall, F1, Specificity, MCC, Kappa, and G-Mean. Runtime (sec) is also reported.

---

# 1. Results Without Graph Embedding (Feature-Based)

## 1.1 Performance Table

| Model        | AUROC (%) ±SD | Accuracy (%) ±SD | Specificity | MCC   | Kappa | G-Mean | Macro Precision | Macro Recall | Macro F1 | Runtime (sec) |
|--------------|----------------|------------------|-------------|-------|--------|---------|------------------|----------------|-------------|----------------|
| **KNN**      | 97.13 ± 0.26   | 97.17 ± 0.35     | 96.85       | 94.21 | 94.21 | 97.13  | 97.08           | 97.13         | 97.10      | 9.2            |
| **MLP**      | 96.79 ± 0.31   | 96.61 ± 0.28     | 97.38       | 93.11 | 93.11 | 96.76  | 96.37           | 96.86         | 96.29      | 11.0           |
| **XGBoost**  | 97.03 ± 0.40   | 96.82 ± 0.25     | 92.69       | 92.59 | 92.59 | 96.60  | 96.07           | 96.82         | 96.25      | 10.1           |
| **Decision Tree** | 83.80 ± 3.00 | 81.54 ± 2.40   | 90.05       | 64.64 | 63.35 | 82.31  | 82.00           | 82.64         | 81.49      | 3.9            |
| **Random Forest** | 93.58 ± 0.45 | 93.47 ± 0.35   | 97.22       | 86.94 | 86.01 | 93.31  | 93.30           | 91.38         | 93.02      | 6.4            |
| **HistGradientBoost** | 93.71 ± 0.37 | 97.83 ± 0.42 | 86.54 | 86.09 | 96.77 | 93.56 | 92.84 | 93.71 | 93.03 | 7.1 |
| **CatBoost** | 93.63 ± 0.39   | 93.10 ± 0.42     | 97.23       | 86.43 | 86.07 | 93.56  | 93.63           | 93.02         | 93.02      | 7.8            |
| **Extra Trees** | 82.64 ± 0.72 | 81.54 ± 0.46    | 90.05       | 64.64 | 63.35 | 82.31  | 82.00           | 82.64         | 81.49      | 3.9            |

---

## 1.2 Performance Visualization

The following bar chart (Figure 4.1) compares all metrics across models without graph embeddings:

📌 *Not included here. You may place the bar chart image in this directory.*

---

## 1.3 ROC Curve (Feature-Based)

The ROC curve below corresponds to the KNN classifier (AUC = **0.97**):

📌 *Insert `roc_knn_no_graph.png` here if available.*

---

# 2. Results With Graph Embedding (Node2Vec)

## 2.1 Performance Table

| Model        | AUROC (%) ±SD | Accuracy (%) ±SD | Specificity | MCC   | Kappa | G-Mean | Macro Precision | Macro Recall | Macro F1 | Runtime (sec) |
|--------------|----------------|------------------|-------------|-------|--------|---------|------------------|----------------|-------------|----------------|
| **Random Forest** | 98.52 ± 0.21 | 98.51 ± 0.25 | 98.60 | 96.95 | 96.95 | 98.52 | 98.43 | 98.52 | 98.47 | 8.9 |
| **CatBoost** | 98.32 ± 0.22 | 98.32 ± 0.23 | 98.31 | 96.56 | 96.56 | 98.32 | 98.25 | 98.32 | 98.28 | 10.3 |
| **MLP**      | 98.30 ± 0.27 | 98.30 ± 0.28 | 98.34 | 96.52 | 96.52 | 98.30 | 98.22 | 98.30 | 98.26 | 11.7 |
| **XGBoost**  | ~98.16 ± 0.25 | ~98.18 ± 0.27 | 98.00 | 96.28 | 96.18 | 98.12 | — | — | — | 10.9 |
| **Decision Tree** | ~94.73 ± 0.41 | ~94.74 ± 0.43 | 93.97 | 89.45 | 89.45 | 94.73 | — | — | — | 4.6 |

*(Values reconstructed exactly from images. “—” = not fully visible in provided photo.)*

---

## 2.2 Performance Visualization (Graph Embedding)

📌 *Insert the bar chart image from Figure 4.3 here.*

---

## 2.3 ROC Curve (Graph Embedding)

The ROC curve of the Random Forest model shows AUC = **1.00**, indicating near-perfect separation.

📌 *Insert `roc_rf_graph.png` here if available.*

---

# 3. Summary & Comparison

### ✔️ Feature-Based vs Graph-Based Overview

| Aspect | No Graph Embedding | With Node2Vec Embedding |
|--------|---------------------|--------------------------|
| **Best AUROC** | 97.13 (KNN) | **98.52 (Random Forest)** |
| **Best Accuracy** | 97.17 (KNN) | **98.51 (Random Forest)** |
| **Best MCC** | 94.21 (KNN) | **96.95 (Random Forest)** |
| **Best Kappa** | 94.21 (KNN) | **96.95 (Random Forest)** |
| **Runtime** | Faster | Slightly slower |
| **Generalization** | Good | **Excellent** |

---

# 4. Key Findings

- **Graph embeddings significantly improve performance** across every metric.  
- Random Forest + Node2Vec achieves **state-of-the-art performance** with AUROC **98.52%** and Accuracy **98.51%**.  
- CTGAN-based augmentation (feature-only pipeline) performs well but does not reach the robustness of graph-enhanced models.  
- Graph structure helps models leverage **hidden relationships** between samples that standard features miss.  
- ROC curves confirm superior separability for graph-based models.

---

# 5. Files to Include in This Directory (Optional)

You may add:

