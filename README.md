# Active Learning Demo 🤖🔍

This repo walks through a simple **active learning** workflow on the Breast Cancer dataset (malignant vs. benign). Instead of labeling everything, we iteratively query the most informative samples to boost performance quickly.

---

## What’s Inside

- **Data & Setup**  
  - Loads the Breast Cancer dataset from `sklearn.datasets`.  
  - Uses `modAL` (with scikit-learn) for active learning utilities.  
  - Standard packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `modAL`.

- **Core Script (`adva.py`)**  
  1. **Initial Seed**  
     - Randomly picks 3 labeled examples to start (`X_train`, `y_train`).  
     - The rest (`X_pool`, `y_pool`) remain “unlabeled.”  
  2. **ActiveLearner Setup**  
     - Base estimator: `RandomForestClassifier(n_estimators=10)`.  
     - Initialized on the 3 seed points to simulate an “oracle” labeling process.  
  3. **Baseline Accuracy**  
     - Evaluate on the full dataset before any queries → print baseline score.  
     - Plot PCA (2D) of all points, colored by correct/incorrect classification. 📊
  4. **Query Loop (40 Iterations)**  
     ```python
     for i in range(40):
         query_idx, query_inst = learner.query(X_pool)               # Uncertainty sampling
         X_new, y_new = X_pool[query_idx].reshape(1, -1), y_pool[query_idx]
         learner.teach(X=X_new, y=y_new)                              # “Label” the queried point
         X_pool, y_pool = np.delete(X_pool, query_idx, 0), np.delete(y_pool, query_idx)
         acc = learner.score(X, y)                                    # Evaluate on all data
         print(f"Accuracy after query {i+1}: {acc:.4f}")
     ```
     - At each step, the most uncertain point is labeled and added to the model.  
     - Accuracy on the full dataset is tracked in `performance_history`.  

- **Plots & Outputs**  
  - **PCA Scatter**: visualizes initial correct vs. incorrect predictions.  
  - **Accuracy Prints**: shows how model accuracy improves as queries accumulate.  

---