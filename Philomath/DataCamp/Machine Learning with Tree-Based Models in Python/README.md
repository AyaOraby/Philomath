# Decision Trees and Ensemble Methods in Machine Learning

## 1. Classification and Regression Trees (CART)

### What are Decision Trees?

Decision trees are like flowcharts that help make decisions based on yes/no questions. They're used in *supervised learning* for:

- **Classification** (e.g., Spam vs Not Spam)
- **Regression** (e.g., Predict house price)

### Key Concepts:

- **Classification Tree**: Predicts a category (label).
- **Regression Tree**: Predicts a continuous value.

### Impurity Measures:

- **Entropy**: Measures the level of surprise or disorder.
- **Gini Index**: Measures the chance of a random sample being misclassified.

Both help the tree decide where to split the data for better accuracy.

### Code Example: Train a Classification Tree

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=3, criterion='gini')
model.fit(X, y)

plt.figure(figsize=(10,6))
plot_tree(model, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
plt.show()
```

---

## 2. The Bias-Variance Tradeoff

### Definitions:

- **Bias**: Model is too simple and underfits the data.
- **Variance**: Model is too complex and overfits the data.

### Tradeoff:

Ideal models strike a balance between bias and variance.

### Ensemble Learning:

Combines multiple models to make better predictions:

- Reduces variance.
- More robust than individual models.

### Code Example: Train vs CV Error

```python
from sklearn.model_selection import cross_val_score
import numpy as np

train_scores = []
cv_scores = []

depths = range(1, 11)
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X, y)
    train_scores.append(model.score(X, y))
    cv_scores.append(np.mean(cross_val_score(model, X, y, cv=5)))

plt.plot(depths, train_scores, label='Training Accuracy')
plt.plot(depths, cv_scores, label='Cross-Validation Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Bias-Variance Tradeoff')
plt.show()
```

---

## 3. Bagging and Random Forests

### Bagging (Bootstrap Aggregating):

- Trains multiple models on random subsets of data.
- Combines predictions through averaging or majority voting.

### Random Forest:

- A collection of decision trees.
- Adds randomness to feature selection at each split.
- More diverse, less prone to overfitting.

### Out-Of-Bag (OOB) Evaluation:

- Samples not used in training a specific tree are used to validate it.
- Helps estimate model performance without separate test data.

### Code Example: Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X, y)

print("OOB Score:", rf.oob_score_)
```

---

## 4. Boosting

### Overview:

- Trains models *sequentially*.
- Each model focuses on the errors of the previous one.

### Types:

- **AdaBoost**: Adjusts weights on samples to focus on difficult cases.
- **Gradient Boosting**: Minimizes errors using gradient descent-like approach.
- **Stochastic Gradient Boosting (SGB)**: Adds randomness to improve generalization.

### Code Example: AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X, y)
print("AdaBoost Accuracy:", ada.score(X, y))
```

---

## 5. Model Tuning

### What is Hyperparameter Tuning?

- **Hyperparameters**: Set before training (e.g., tree depth).
- **Parameters**: Learned from data (e.g., regression weights).

### Grid Search Cross-Validation:

- Tests multiple combinations of hyperparameters.
- Uses cross-validation to select the best set.

### Common Tree Hyperparameters:

- `max_depth`
- `min_samples_split`
- `n_estimators` (for ensemble models)

### Code Example: Grid Search with DecisionTreeClassifier

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 5, 10]
}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
gs.fit(X, y)

print("Best Parameters:", gs.best_params_)
print("Best CV Score:", gs.best_score_)
```

---

## Summary Table

| Concept                    | What It Does                                               |
| -------------------------- | ---------------------------------------------------------- |
| **CART**                   | Simple tree model for classification and regression        |
| **Bias-Variance Tradeoff** | Balances underfitting (bias) and overfitting (variance)    |
| **Bagging**                | Builds many models in parallel and averages them           |
| **Random Forest**          | A bagging model with added randomness in feature selection |
| **Boosting**               | Builds models in sequence to improve on previous ones      |
| **Model Tuning**           | Helps find the best model settings for top performance     |

---



