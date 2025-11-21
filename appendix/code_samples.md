
# Code Samples

This appendix contains selected code excerpts used in the implementation of the two experimental pipelines.

--------------------------------------------------------------------

1. Load and Encode Dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/labeled.combined.csv")
le = LabelEncoder()
data["label"] = le.fit_transform(data["label"])

--------------------------------------------------------------------

2. Feature Scaling (Common to Both Pipelines)

from sklearn.preprocessing import RobustScaler

exclude_columns = ["label"]
excluded_data = data[exclude_columns]
scaling_data = data.drop(columns=exclude_columns)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(scaling_data)

--------------------------------------------------------------------

3. k-NN Graph Construction and Node2Vec (Pipeline A)

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
embeddings = [model.wv[str(i)] for i in range(len(G.nodes))]

--------------------------------------------------------------------

4. CTGAN Synthetic Data Generation (Pipeline B)

from ctgan import CTGAN

loaded = ctgan.load("models/js_train.pkl")
samples = loaded.sample(16800)
samples = samples[samples["label"] == 0]

# Combine with real training data
augmented_df = pd.concat([train, samples])

--------------------------------------------------------------------

5. Model Training and Evaluation (Common)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

model.fit(X_train, y_train)
predicted = model.predict(X_test)

acc = accuracy_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
prec = precision_score(y_test, predicted)
rec = recall_score(y_test, predicted)
auc = roc_auc_score(y_test, predicted)

--------------------------------------------------------------------

6. ROC Curve Plotting

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label=f"Model (AUC={auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

--------------------------------------------------------------------

These code samples represent the key components of the experimental pipelines used in the thesis.
