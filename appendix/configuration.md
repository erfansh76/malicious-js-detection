# Experimental Configuration

This appendix summarizes the software, data, and experimental settings required to reproduce the results.

1. Software Environment
- Python: 3.x
- Package manager: pip

Install all dependencies:
pip install -r requirements.txt

Main libraries used:
numpy
pandas
matplotlib
scikit-learn
networkx
node2vec
ctgan
lightgbm
catboost
xgboost
imbalanced-learn

2. Dataset
- Input file: data/labeled.combined.csv
- Task: binary classification of JavaScript files
- Target column: label (encoded 0/1)

Load dataset:
import pandas as pd
data = pd.read_csv("data/labeled.combined.csv")

3. Preprocessing

3.1 Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

3.2 Feature Scaling
from sklearn.preprocessing import RobustScaler
exclude_columns = ["label"]
excluded_data = data[exclude_columns]
scaling_data = data.drop(columns=exclude_columns)
scaler = RobustScaler()
scaled_data = scaler.fit_transform(scaling_data)

3.3 Reconstruct Final Dataframe
import pandas as pd
scaled_df = pd.DataFrame(scaled_data, columns=scaling_data.columns)
final_df = pd.concat([scaled_df, excluded_data.reset_index(drop=True)], axis=1)

3.4 Train/Test Split
from sklearn.model_selection import StratifiedKFold
X = final_df.drop("label", axis=1)
y = final_df["label"]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

4. Graph Construction and Node2Vec (Pipeline A)
from sklearn.neighbors import kneighbors_graph
from node2vec import Node2Vec
import networkx as nx

adj_matrix = kneighbors_graph(
    X,
    n_neighbors=10,
    mode="connectivity",
    include_self=False,
    n_jobs=-1
)

G = nx.from_scipy_sparse_array(adj_matrix)

node2vec = Node2Vec(
    G,
    dimensions=64,
    walk_length=30,
    num_walks=5,
    p=1.0,
    q=1.0,
    workers=8
)

model = node2vec.fit(window=5)

5. CTGAN Configuration (Pipeline B)
from ctgan import CTGAN
loaded = ctgan.load("models/js_train.pkl")
samples = loaded.sample(16800)
samples = samples[samples["label"] == 0]
# Real + synthetic samples combined

6. Models and Evaluation

Classifiers used:
SVM
Random Forest
Extra Trees
Gradient Boosting
HistGradientBoosting
LightGBM
XGBoost
CatBoost
Decision Tree
KNN
MLP
Logistic Regression
SGDClassifier
Gaussian Naive Bayes
LDA / QDA

Metrics computed:
AUROC
Accuracy
Precision
Recall
F1-score
Specificity
MCC
Cohen’s Kappa
Macro/Weighted Precision, Recall, F1
Geometric Mean

Results are averaged across folds and reported as mean ± standard deviation.
