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

---

## 4. Boosting

### Overview:
- Trains models *sequentially*.
- Each model focuses on the errors of the previous one.

### Types:
- **AdaBoost**: Adjusts weights on samples to focus on difficult cases.
- **Gradient Boosting**: Minimizes errors using gradient descent-like approach.
- **Stochastic Gradient Boosting (SGB)**: Adds randomness to improve generalization.

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

---

## Summary Table

| Concept                     | What It Does                                               |
|----------------------------|------------------------------------------------------------|
| **CART**                   | Simple tree model for classification and regression         |
| **Bias-Variance Tradeoff** | Balances underfitting (bias) and overfitting (variance)     |
| **Bagging**                | Builds many models in parallel and averages them            |
| **Random Forest**          | A bagging model with added randomness in feature selection  |
| **Boosting**               | Builds models in sequence to improve on previous ones       |
| **Model Tuning**           | Helps find the best model settings for top performance      |

---

Would you like to add Python code examples or diagrams to this file?

