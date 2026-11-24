# GitHub Developer Classification | Graph Machine Learning and Ensemble Models

<div align="center">

<img src="./github_graph_ml.png" alt="GitHub Developer Classification — Graph Machine Learning and Ensemble Models" width="750">

*Graph Machine Learning — Developer network analysis, feature engineering, and classification using ensemble models.*

</div>

### Network Topology, Feature Engineering, Supervised Classification, and Ensemble Learning with Python

**Project Type:** Graph Machine Learning Research  
**Field:** Data Science | Machine Learning | Graph Theory | Network Analytics  
**Programming Language:** Python  
**Environment:** Google Colab | Jupyter Notebook  
**Author:** Data Engineer Luis Barahona

---

## 1. Project Overview

This project explores Graph Machine Learning techniques for classifying GitHub developers using information derived from their network relationships and node attributes.

The notebook uses the MUSAE GitHub network dataset to investigate whether graph topology and numerical features can distinguish developer categories associated with machine learning and web development.

Rather than training a Graph Neural Network, the project follows a feature-based Graph Machine Learning approach.

Network metrics are extracted using NetworkX and combined with node attributes to train supervised classification models.

The experimental workflow includes:

- Graph data preprocessing.
- Topological feature extraction.
- Centrality analysis.
- Link-based similarity metrics.
- Node feature transformation.
- Dataset integration.
- Stratified train-validation-test splitting.
- Hyperparameter optimization.
- Support Vector Machine classification.
- Gradient boosting models.
- Stacking ensembles.
- ROC-AUC and classification evaluation.

The project follows a CRISP-DM-inspired analytical structure.

---

## 2. Research Objectives

The main objectives are to:

1. Load and inspect the MUSAE GitHub network dataset.
2. Construct a graph representing developer relationships.
3. Clean node and edge information.
4. Calculate graph centrality metrics.
5. Explore local graph similarity measures.
6. Transform node attributes into tabular features.
7. Integrate graph metrics with developer labels.
8. Divide observations into training, validation, and testing datasets.
9. Train multiple machine learning classifiers.
10. Optimize hyperparameters through cross-validation.
11. Compare classification performance.
12. Explore whether ensemble learning improves model evaluation.

---

## 3. Dataset Description

The project uses three CSV files.

```text
musae_git_edges.csv
musae_git_features.csv
musae_git_target.csv
```

### 3.1 Edge Dataset

**File:** `musae_git_edges.csv`

This file contains pairs of developer node identifiers.

| Column | Description |
|:---|:---|
| id_1 | First node identifier |
| id_2 | Second node identifier |

The saved execution reports:

**289,003 edges**

The notebook removes duplicate edges and self-loops during preprocessing.

The saved dataset size remains 289,003 after these operations.

### 3.2 Node Features

**File:** `musae_git_features.csv`

The dataset contains node-feature associations.

| Column | Description |
|:---|:---|
| node | Developer node identifier |
| feature | Associated numerical feature identifier |

The saved execution reports:

**690,374 node-feature records**

These records correspond to 37,700 developers.

### 3.3 Developer Labels

**File:** `musae_git_target.csv`

The target dataset contains:

| Column | Description |
|:---|:---|
| id | Developer node identifier |
| name | GitHub username in the original dataset |
| ml_target | Binary developer classification label |

The binary target represents the developer category associated with machine learning.

The dataset supports a classification experiment involving developer profiles and network characteristics.

The notebook removes the username column from the modeling target table.

---

## 4. Data Preparation

The project preprocesses the three source datasets independently.

### Edge Cleaning

The notebook removes duplicate relationships and self-loops.

```python
edges_df = edges_df.drop_duplicates()

edges_df = edges_df[
    edges_df["id_1"] != edges_df["id_2"]
]
```

This produces a cleaned edge table for graph construction.

### Node Feature Transformation

The original node-feature table is transformed into a wide representation.

```python
features_wide = (
    features_df
    .assign(
        dim=lambda df:
        df.groupby("node").cumcount()
    )
    .pivot(
        index="node",
        columns="dim",
        values="feature"
    )
    .fillna(0)
)
```

The resulting columns are named:

```text
emb_0
emb_1
...
emb_41
```

### Saved Transformation Results

| Stage | Shape |
|:---|:---|
| Original feature table | 690,374 × 2 |
| Transformed feature table | 37,700 × 42 |
| Target table | 37,700 × 1 |

### Representation Limitation

The columns named `emb_*` contain feature identifiers arranged according to their occurrence order for each node.

They are not demonstrated to be learned embeddings.

A stronger representation would encode the feature identifiers consistently across all developers, for example using sparse multi-hot features.

### Target Preparation

The label field is renamed:

```python
target_df = (
    target_df
    .drop(columns=["name"])
    .drop_duplicates(subset="id")
    .rename(
        columns={"ml_target": "label"}
    )
    .set_index("id")
)
```

### Data Integrity

The saved notebook reports:

**37,700 nodes shared by the edge, feature, and target datasets.**

This confirms that the main source tables can be aligned by node identifier in the recorded experiment.

---

## 5. Graph Construction

NetworkX is used to construct the network.

```python
import networkx as nx

G = nx.from_pandas_edgelist(
    edges_df,
    source="id_1",
    target="id_2",
    create_using=nx.DiGraph()
)
```

The notebook models the imported edges as directed connections.

Some local similarity metrics are subsequently calculated on an undirected version of this graph.

### Important Consideration

The direction of the imported edges should be checked against the original dataset definition.

A computationally directed representation does not independently establish that the underlying relationships have a meaningful direction.

---

## 6. Topological Feature Engineering

The notebook extracts numerical features describing each node's structural position within the network.

### Degree

Counts the connections associated with a node.

### In-Degree

Counts incoming edges in the directed graph representation.

### Out-Degree

Counts outgoing edges.

### Degree Centrality

Measures node connectivity relative to the graph size.

### Betweenness Centrality

Measures how frequently a node lies along shortest paths between other nodes.

### Eigenvector Centrality

Measures structural importance based on connections with other important nodes.

### PageRank

Measures node importance through an iterative network-ranking algorithm.

### Clustering Coefficient

Measures how connected a node's neighbors are to each other.

### Average Neighbor Degree

Summarizes the connectivity of neighboring nodes.

### Python Example

```python
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
degree = dict(G.degree())

degree_centrality = nx.degree_centrality(G)

pagerank = nx.pagerank(G)

clustering = nx.clustering(
    G.to_undirected()
)

avg_neighbor_degree = (
    nx.average_neighbor_degree(G)
)
```

### Betweenness Approximation

The notebook approximates betweenness centrality using 500 sampled nodes.

```python
betweenness = nx.betweenness_centrality(
    G,
    k=500,
    normalized=True,
    seed=42
)
```

This reduces the computational burden compared with evaluating all node pairs.

The resulting values are approximations rather than exact betweenness scores.

---

## 7. Eigenvector Centrality

The notebook checks whether the graph is strongly connected.

The saved output indicates:

```text
Graph is not strongly connected.
Calculating eigenvector centrality per SCC.
```

The implementation calculates eigenvector centrality separately within strongly connected components when possible.

Singleton components receive a default value of zero.

### Limitation

Component-wise eigenvector scores are not necessarily directly comparable across different strongly connected components.

This matters when using the values as features in a global classifier.

Alternative centrality definitions and normalization approaches should be considered in future experiments.

---

## 8. Link-Based Similarity Features

The notebook explores several graph similarity metrics.

### 8.1 Common Neighbors

Counts the shared neighbors of two nodes.

```text
CN(A, B) = |N(A) ∩ N(B)|
```

### 8.2 Jaccard Coefficient

Measures the overlap between neighboring node sets.

```text
J(A, B) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
```

### 8.3 Adamic-Adar Index

Weights shared neighbors according to their connectivity.

```text
AA(A, B) =
Sum over common neighbors w of 1 / log(degree(w))
```

### 8.4 Preferential Attachment

Estimates a potential connection score based on node degrees.

```text
PA(A, B) = degree(A) × degree(B)
```

### Implementation

The notebook calculates these measures manually using Python sets and dictionaries.

The calculations are demonstrated on existing network edges.

### Important Distinction

These experiments demonstrate link similarity calculations.

The completed classification table does not demonstrate a separate, validated link-prediction model.

The final 51 input features consist of 42 transformed node features and nine topological features.

The planned link-metric aggregation is not demonstrated as part of the final feature matrix.

---

## 9. Final Feature Matrix

The notebook combines:

- Nine topological features.
- Forty-two transformed node-feature columns.
- One binary label.

The final supervised learning input contains:

**51 numerical features per developer.**

### Dataset Split

A stratified 60/20/20 split is implemented.

| Dataset | Developers | Percentage |
|:---|---:|---:|
| Training | 22,620 | 60% |
| Validation | 7,540 | 20% |
| Testing | 7,540 | 20% |
| Total | 37,700 | 100% |

### Python Implementation

```python
X_train, X_rem, y_train, y_rem = (
    train_test_split(
        X,
        y,
        train_size=0.60,
        stratify=y,
        random_state=42
    )
)

X_val, X_test, y_val, y_test = (
    train_test_split(
        X_rem,
        y_rem,
        train_size=0.50,
        stratify=y_rem,
        random_state=42
    )
)
```

Stratification helps preserve the target-class distribution across the three subsets.

---

## 10. Machine Learning Models

The project experiments with six classification approaches.

### 10.1 Linear Support Vector Machine

LinearSVC is used as a linear classification baseline.

A pipeline combines:

- StandardScaler.
- LinearSVC.

Hyperparameter optimization is performed using GridSearchCV.

### 10.2 XGBoost

XGBoost uses gradient-boosted decision trees.

The notebook explores:

- Number of estimators.
- Maximum tree depth.
- Learning rate.
- Row sampling.
- Column sampling.

### 10.3 LightGBM

LightGBM provides a gradient boosting implementation designed for efficient tabular learning.

The notebook evaluates multiple combinations of:

- Number of estimators.
- Number of leaves.
- Learning rate.
- Feature fraction.

### 10.4 CatBoost

CatBoost is evaluated as another gradient boosting approach.

The notebook explores:

- Iterations.
- Tree depth.
- Learning rate.
- L2 regularization.

### 10.5 Histogram-Based Gradient Boosting

A HistGradientBoostingClassifier is trained after using a Random Forest-based feature selection procedure.

The experiment uses a reduced training sample and selected input features.

### 10.6 Stacking Ensemble

The final experiment combines three base learners:

- HistGradientBoostingClassifier.
- LightGBM.
- CatBoost.

A Logistic Regression model acts as the final meta-classifier.

The objective is to explore whether combining multiple models improves predictive performance.

---

## 11. Hyperparameter Optimization

The LinearSVC and XGBoost experiments use GridSearchCV with five-fold cross-validation.

### LinearSVC

The saved output reports:

```text
Best C: 10
Best loss: squared_hinge
Best dual: False
```

**Cross-validation ROC-AUC: 0.6440**

### XGBoost

The saved output reports:

```text
n_estimators: 300
max_depth: 3
learning_rate: 0.1
subsample: 1.0
colsample_bytree: 0.8
```

**Cross-validation ROC-AUC: 0.8612**

### Interpretation

XGBoost achieved the higher cross-validation ROC-AUC in the two tuned experiments.

However, cross-validation and holdout validation scores are different evaluation procedures and should be distinguished when comparing models.

---

## 12. Experimental Results

The following values are taken from the notebook's saved execution outputs.

| Model | Evaluation | ROC-AUC |
|:---|:---|---:|
| LinearSVC | 5-fold cross-validation | 0.6440 |
| XGBoost | 5-fold cross-validation | 0.8612 |
| LightGBM | Validation set | 0.8550 |
| CatBoost | Validation set | 0.8507 |
| HistGradientBoosting | Validation set | 0.8302 |
| Stacking Ensemble | Validation set | 0.8408 |

### Key Observations

- XGBoost obtained a cross-validation ROC-AUC of 0.8612.
- LightGBM obtained a validation ROC-AUC of approximately 0.8550.
- CatBoost obtained a validation ROC-AUC of approximately 0.8507.
- The stacking model obtained a validation ROC-AUC of 0.8408.
- The stacking configuration did not demonstrate an improvement over the best recorded XGBoost cross-validation score.

Because the scores come from different evaluation procedures, they do not establish a definitive model ranking on an identical held-out test set.

---

## 13. HistGradientBoosting Classification Report

The notebook provides a more detailed validation report for HistGradientBoosting.

### Recorded Results

| Metric | Value |
|:---|---:|
| Accuracy | 82.94% |
| ROC-AUC | 0.8302 |
| Class 0 precision | 0.8494 |
| Class 0 recall | 0.9360 |
| Class 0 F1-score | 0.8906 |
| Class 1 precision | 0.7402 |
| Class 1 recall | 0.5236 |
| Class 1 F1-score | 0.6133 |

### Class Distribution

| Class | Validation Observations |
|:---|---:|
| 0 | 5,592 |
| 1 | 1,948 |
| Total | 7,540 |

### Interpretation

The classifier performs better on class 0 than class 1.

The class 1 recall of approximately 52.36% indicates that a substantial proportion of positive examples are not correctly detected.

This highlights the importance of evaluating class-specific metrics rather than overall accuracy alone.

---

## 14. Stacking Ensemble Experiment

The final implementation uses Scikit-learn's StackingClassifier.

### Base Learners

- Histogram Gradient Boosting.
- LightGBM.
- CatBoost.

### Meta-Learner

- Logistic Regression.

### Python Structure

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[
        ("hgb", hgb),
        ("lgb", lgb),
        ("cbc", cbc)
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1,
    passthrough=False
)
```

### Saved Result

**Validation ROC-AUC: 0.8408**

This demonstrates the execution of an ensemble classification workflow.

However, the result does not establish that stacking improves performance over individually tuned gradient boosting models.

---

## 15. Important Evaluation Limitation

The notebook prepares a separate test dataset containing 7,540 developers.

However, the saved execution does not include a final evaluation of the selected model on this untouched test set.

The concluding Markdown also contains unfinished performance placeholders.

Therefore:

- The recorded cross-validation results are available.
- Several holdout validation scores are available.
- A final independent test benchmark is not documented.
- The model-selection conclusion remains preliminary.

A future version should select one model using the training and validation information, freeze the configuration, and evaluate it once on the reserved test dataset.

---

## 16. Methodological Considerations

### Graph-Wide Feature Extraction

The notebook calculates network metrics using the full graph before splitting developer labels.

This is compatible with some transductive learning settings, where the entire graph structure is assumed to be known.

However, it does not represent a strict inductive setting in which completely unseen developers and relationships arrive after model training.

### Feature Representation

The `emb_*` columns are ordered feature identifiers, not verified learned graph embeddings.

Their meaning and consistency should be reviewed.

### Graph Direction

The graph is represented as directed, although the intended meaning of edge direction should be verified.

### Centrality Computation

Approximate betweenness and component-wise eigenvector centrality involve implementation choices that can influence features.

### Link Metrics

Link similarity calculations are demonstrated separately but are not included in the final 51-feature classification input.

### Model Comparison

The notebook does not report every classifier using an identical final test protocol.

### Class Imbalance

The positive class is less frequent than the negative class.

Class-specific recall and F1-score should therefore remain central evaluation metrics.

---

## 17. Technologies Used

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| Pandas | Data preparation |
| NumPy | Numerical calculations |
| NetworkX | Graph modeling and topology |
| python-igraph | Additional graph analysis dependency |
| Scikit-learn | Modeling pipelines, validation, and metrics |
| LinearSVC | Linear classification |
| XGBoost | Gradient boosting classification |
| LightGBM | Efficient gradient boosting |
| CatBoost | Ensemble tree-based classification |
| Random Forest | Feature selection |
| HistGradientBoosting | Histogram-based ensemble classification |
| StackingClassifier | Ensemble integration |
| Matplotlib | Visualization |
| Google Colab | Cloud execution environment |
| Jupyter Notebook | Interactive experimentation |

---

## 18. How to Run

### Google Colab

[Open the Original Notebook](https://colab.research.google.com/drive/14atHD40Qu-CbjEBMWbIrs4TcRmSTEHxG)

### Required Datasets

```text
musae_git_edges.csv
musae_git_features.csv
musae_git_target.csv
```

These files must be uploaded to the notebook's working directory.

### Install Dependencies

```bash
pip install pandas numpy networkx matplotlib scikit-learn xgboost lightgbm catboost python-igraph notebook
```

### Execution Steps

1. Open the notebook in Google Colab.
2. Install the dependencies.
3. Upload the three MUSAE dataset files.
4. Load the graph edges and node features.
5. Clean the source datasets.
6. Construct the network.
7. Calculate node centrality metrics.
8. Explore local graph similarity scores.
9. Create the final feature matrix.
10. Perform stratified data splitting.
11. Train the LinearSVC model.
12. Tune XGBoost using cross-validation.
13. Evaluate LightGBM and CatBoost.
14. Explore feature selection and HistGradientBoosting.
15. Train the stacking ensemble.
16. Review the recorded validation results.

### Performance Considerations

The network contains approximately 37,700 nodes and 289,003 edges.

Some graph computations and hyperparameter searches may require substantial memory and processing time.

The notebook performs multiple expensive GridSearchCV experiments.

A high-memory environment may be helpful.

---

## 19. Future Improvements

Potential next steps include:

- Correcting the node-feature representation.
- Validating edge direction and graph semantics.
- Creating sparse multi-hot node features.
- Adding reliable graph embedding techniques.
- Evaluating all models on the same validation protocol.
- Selecting a final model before test evaluation.
- Reporting test ROC-AUC, accuracy, precision, recall, and F1.
- Adding confusion matrices and ROC curves.
- Comparing computational training costs.
- Testing model behavior under inductive splits.
- Exploring GraphSAGE and Graph Attention Networks.
- Evaluating actual graph neural network architectures.
- Adding automated preprocessing and validation tests.
- Creating reproducible experiment configurations.

---

## 20. Potential Applications

The techniques explored in this project may support:

- Graph Machine Learning research.
- Developer network analytics.
- Social network classification.
- Community and relationship analysis.
- Recommendation system research.
- Graph feature engineering.
- Organizational collaboration analysis.
- Machine learning model benchmarking.

These are potential uses of the methods, not independently validated production capabilities of this implementation.

---

## 21. Project Resources

### Google Colab Notebook

[Open GitHub Developer Classification Notebook](https://colab.research.google.com/drive/14atHD40Qu-CbjEBMWbIrs4TcRmSTEHxG)

### NetworkX

[NetworkX Documentation](https://networkx.org/documentation/stable/)

### Scikit-learn

[Scikit-learn Documentation](https://scikit-learn.org/stable/)

### XGBoost

[XGBoost Documentation](https://xgboost.readthedocs.io/)

### LightGBM

[LightGBM Documentation](https://lightgbm.readthedocs.io/)

### CatBoost

[CatBoost Documentation](https://catboost.ai/)

### Dataset Attribution

The project uses the externally sourced MUSAE GitHub network dataset.

The original dataset authors and distribution terms should be acknowledged when publishing the raw files or derived datasets.

---

## 22. Conclusion

This project explores the combination of graph topology, feature engineering, and supervised machine learning for GitHub developer classification.

Using Python and NetworkX, the notebook constructs a network containing 37,700 developers and 289,003 edges, and prepares a 51-feature classification dataset.

Multiple classifiers are evaluated, including LinearSVC, XGBoost, LightGBM, CatBoost, HistGradientBoosting, and stacking ensembles.

XGBoost achieves a recorded five-fold cross-validation ROC-AUC of 0.8612, while LightGBM records a validation ROC-AUC of approximately 0.8550.

The ensemble experiment demonstrates the integration of three gradient boosting approaches, although it does not establish an improvement over the strongest individual results.

A complete final test-set evaluation and improved feature representation are important next steps.

Overall, the project provides a practical foundation for Graph Machine Learning, ensemble classification, network feature engineering, and reproducible model evaluation.

---

## Author

**Data Engineer Luis Barahona**

Mexican Data Engineer and 2026 graduate of Universidad del Caribe, interested in Data Engineering, Data Science, Artificial Intelligence, Graph Machine Learning, and continuous professional development.

---

## Contact

| Platform | Contact |
|:---|:---|
| Email | [190300395@ucaribe.edu.mx](mailto:190300395@ucaribe.edu.mx) |
| GitHub | [github.com/LuisBarahonaMX](https://github.com/LuisBarahonaMX) |
| LinkedIn | [linkedin.com/in/luis-barahona-mx](https://www.linkedin.com/in/luis-barahona-mx/) |
| Instagram | [@luisbarahonamx](https://www.instagram.com/luisbarahonamx/) |
| X | [@luisbarahonaMX](https://x.com/luisbarahonaMX) |
| Website | [SIKO AI Platform](https://sikoai.web.app/) |
