# Machine Learning with PySpark

## Introduction
Apache Spark is a powerful distributed computing framework designed for big data processing. PySpark is the Python API for Spark, enabling scalable machine learning by distributing computations across multiple nodes. It is widely used for handling large-scale data efficiently and supports multiple machine learning algorithms.

### Why Use PySpark for Machine Learning?
- **Scalability**: Handles large datasets efficiently across distributed clusters.
- **Distributed Computing**: Parallel processing speeds up computations.
- **In-memory Computation**: Reduces I/O latency, improving performance.
- **Integration with Big Data Tools**: Works seamlessly with Hadoop, HDFS, and cloud-based solutions.

---

## Setting Up PySpark
To use PySpark, first, install it using pip:
```sh
pip install pyspark
```
Then, create a `SparkSession`, which is the entry point for working with Spark:
```python
from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName("Machine Learning with PySpark").getOrCreate()
```

### Loading Data
PySpark supports multiple file formats such as CSV, JSON, and Parquet.
```python
# Load dataset from a CSV file
iris_df = spark.read.csv("iris.csv", header=True, inferSchema=True)
print("✅ Dataset Loaded")

# Display first 5 rows of the dataset
iris_df.show(5)
```

---

## Data Preprocessing
Before training models, data preprocessing is essential. It includes handling missing values, encoding categorical variables, and assembling feature vectors.

### Handling Categorical Data
Many machine learning models require numerical inputs. We use `StringIndexer` to convert categorical labels into numerical indices.
```python
from pyspark.ml.feature import StringIndexer

# Convert categorical labels into numerical indices
indexer = StringIndexer(inputCol="species", outputCol="label")
iris_df = indexer.fit(iris_df).transform(iris_df)
```

### Feature Vector Assembly
Models require a single column of numerical features. `VectorAssembler` combines multiple columns into a feature vector.
```python
from pyspark.ml.feature import VectorAssembler

# Combine feature columns into a single vector column
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], outputCol="features")
iris_df = assembler.transform(iris_df)
```

---

## Building a Machine Learning Pipeline
A `Pipeline` automates the transformation and modeling process by chaining multiple steps together.
```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

# Define logistic regression model
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Create a pipeline with data transformations and model
pipeline = Pipeline(stages=[indexer, assembler, lr])

# Train the model
pipeline_model = pipeline.fit(iris_df)
print("✅ Pipeline Created and Trained")
```

---

## Classification Models
Classification predicts categorical labels based on input features.

### Decision Tree Classifier
A Decision Tree is a hierarchical model used for classification.
```python
from pyspark.ml.classification import DecisionTreeClassifier

# Initialize and train a Decision Tree model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
model = dt.fit(iris_df)
```

### Logistic Regression
Logistic Regression predicts probabilities of categorical outcomes.
```python
# Initialize and train a Logistic Regression model
lr = LogisticRegression(labelCol="label", featuresCol="features")
model = lr.fit(iris_df)
```

---

## Regression Models
Regression predicts continuous values, such as prices or durations.

### Linear Regression
```python
from pyspark.ml.regression import LinearRegression

# Initialize and train a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(iris_df)
```

### Feature Engineering
New features can be created for better accuracy.
```python
from pyspark.ml.feature import Bucketizer

# Bucketize a continuous variable into discrete bins
bucketizer = Bucketizer(splits=[0, 5, 10, 15], inputCol="sepal_length", outputCol="bucketed_sepal")
iris_df = bucketizer.transform(iris_df)
```

### Regularization
Regularization prevents overfitting.
```python
# Apply L2 regularization to a linear regression model
lr = LinearRegression(featuresCol="features", labelCol="label", regParam=0.1)
model = lr.fit(iris_df)
```

---

## Model Evaluation
Model evaluation measures performance using metrics like accuracy, precision, and recall.
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Define an evaluator to measure model accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Evaluate the trained model
accuracy = evaluator.evaluate(model.transform(iris_df))
print(f"Model Accuracy: {accuracy}")
```

---

## Hyperparameter Tuning
Using `CrossValidator` and `ParamGridBuilder` to optimize model parameters.
```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Define a grid of hyperparameters
param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()

# Apply cross-validation for model tuning
crossval = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator)
model = crossval.fit(iris_df)
```

---

## Advanced Models
### Random Forest Classifier
An ensemble learning method that improves accuracy.
```python
from pyspark.ml.classification import RandomForestClassifier

# Train a Random Forest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
model = rf.fit(iris_df)
```

### Gradient-Boosted Trees
Boosting improves weak models by sequential training.
```python
from pyspark.ml.classification import GBTClassifier

# Train a Gradient-Boosted Trees model
gbt = GBTClassifier(labelCol="label", featuresCol="features")
model = gbt.fit(iris_df)
```

---

## Conclusion
PySpark provides a powerful framework for scalable machine learning, supporting classification, regression, and ensemble methods. Pipelines streamline workflows, while hyperparameter tuning and model evaluation improve performance. With distributed computing, PySpark is ideal for handling large-scale machine learning tasks efficiently.

