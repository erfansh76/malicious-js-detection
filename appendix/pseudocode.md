# Pseudocode of the Experimental Pipelines

This appendix describes the high-level pseudocode of the two experimental pipelines implemented in this project.

---

## 1. Pipeline A — Graph-Based Model (Node2Vec + Machine Learning)

**Source file:** `model_with_graph_embedding.py`

### Steps
1. Load dataset from `data/labeled.combined.csv`.
2. Encode label column using `LabelEncoder`.
3. Separate feature columns and the label column.
4. Apply `RobustScaler` to all feature columns.
5. Reconstruct the final dataframe (scaled features + label).
6. Build a k-NN graph using `kneighbors_graph`.
7. Convert adjacency matrix to a NetworkX graph.
8. Run Node2Vec with:
   - dimensions = 64  
   - walk_length = 30  
   - num_walks = 5  
   - p = 1.0, q = 1.0  
9. Extract node embeddings into a NumPy matrix.
10. Define multiple ML classifiers (RF, SVM, LightGBM, XGBoost, CatBoost, etc.).
11. Perform Stratified K-Fold (5 folds):
    - Train each classifier on training embeddings
    - Evaluate on test embeddings
    - Compute metrics:
      AUROC, Accuracy, Precision, Recall, F1,
      Specificity, MCC, Cohen’s Kappa, Macro/Weighted metrics, G-Mean
12. Calculate mean ± std of metrics.
13. Plot ROC curves for all classifiers.
14. Measure and report execution time.

---

## 2. Pipeline B — Feature-Based + CTGAN Augmentation + Machine Learning

**Source file:** `model_without_graph_embedding.py`

### Steps
1. Load dataset from `data/labeled.combined.csv`.
2. Encode labels using `LabelEncoder`.
3. Scale all numeric features using `RobustScaler`.
4. Reconstruct the final dataframe (scaled features + label).
5. Perform Stratified K-Fold split.
6. Use the first fold as:
   - `X_train`, `X_test`, `y_train`, `y_test`
7. Combine `X_train` and `y_train` into a training dataframe.
8. Load a pre-trained CTGAN model (`js_train.pkl`).
9. Generate synthetic samples.
10. Filter synthetic rows by class (e.g., label = 0).
11. Combine real + synthetic samples (augmented dataframe).
12. Split augmented dataframe into `X_train_gan` and `y_train_gan`.
13. Define the same ML classifiers as Pipeline A.
14. Train each classifier on augmented data.
15. Evaluate on the original test set:
    AUROC, Accuracy, Precision, Recall, F1,
    Specificity, MCC, Kappa, Macro/Weighted metrics, G-Mean
16. Compute aggregated metrics.
17. Plot ROC curves.
18. Measure and report total runtime.
