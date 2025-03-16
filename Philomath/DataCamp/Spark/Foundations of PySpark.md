# Foundations of PySpark

## Description
This course introduces PySpark, the Python package for Apache Spark, a powerful tool for parallel computation with large datasets. You'll learn how to use PySpark to manipulate data, create machine learning pipelines, and optimize model performance. The course is structured around working with flight data from Portland and Seattle to predict flight delays.

## Course Outline

### 1. Getting to Know PySpark
Learn how Spark manages data and how to read and write tables using Python.

#### Introduction to Spark
Apache Spark is an open-source distributed computing framework that enables fast and efficient data processing. It is designed for big data and provides APIs for multiple programming languages, including Python (PySpark).

```python
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("PySparkExample").getOrCreate()
```

#### Using Spark in Python
PySpark allows users to leverage Spark’s capabilities using Python. It provides DataFrames, similar to Pandas but optimized for distributed computing.

```python
data = [("Alice", 29), ("Bob", 31), ("Catherine", 25)]
df = spark.createDataFrame(data, ["Name", "Age"])
df.show()
```

#### Examining The SparkContext
The `SparkContext` represents the entry point to interact with the cluster.

```python
sc = spark.sparkContext
print(sc.version)  # Print Spark version
```

#### Working with DataFrames
DataFrames are distributed collections of data, similar to tables in a database.

```python
df.printSchema()
df.select("Name").show()
```

#### Creating a SparkSession
The `SparkSession` is the entry point for working with structured data in Spark.

```python
spark = SparkSession.builder.appName("MyApp").getOrCreate()
```

#### Viewing Tables
You can create and query tables using Spark SQL.

```python
df.createOrReplaceTempView("people")
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()
```

#### Writing Queries
Use Spark SQL to query structured data efficiently.

```python
result = spark.sql("SELECT Name, Age FROM people WHERE Age > 25")
result.show()
```

#### Converting Spark DataFrames to Pandas
PySpark DataFrames can be converted to Pandas DataFrames for further analysis.

```python
pandas_df = df.toPandas()
print(pandas_df.head())
```

### 2. Manipulating Data
Explore the `pyspark.sql` module for optimized data queries.

#### Creating and Transforming Columns
Add new columns and manipulate data using functions.

```python
from pyspark.sql.functions import col

df = df.withColumn("AgePlusOne", col("Age") + 1)
df.show()
```

#### Filtering Data
Filter data based on conditions.

```python
filtered_df = df.filter(df.Age > 25)
filtered_df.show()
```

#### Selecting Data
Extract specific columns from a DataFrame.

```python
df.select("Name").show()
```

#### Aggregating and Grouping Data
Group and aggregate data using functions like `count()` and `avg()`.

```python
from pyspark.sql.functions import avg

df.groupBy("Name").agg(avg("Age")).show()
```

### 3. Getting Started with Machine Learning Pipelines
Understand how PySpark facilitates machine learning workflows.

#### Machine Learning Pipelines
A pipeline consists of multiple stages such as data transformation and model training.

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["Age"], outputCol="features")
pipeline = Pipeline(stages=[assembler])
```

#### Feature Engineering
Transform categorical data and prepare features.

```python
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="Name", outputCol="NameIndex")
df = indexer.fit(df).transform(df)
df.show()
```

#### Splitting and Transforming Data
Divide data into training and test sets.

```python
train, test = df.randomSplit([0.8, 0.2])
```

### 4. Model Tuning and Selection
Build and evaluate a predictive model for flight delays.

#### Introduction to Logistic Regression
Logistic regression is a classification algorithm commonly used for binary outcomes.

```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label")
```

#### Cross-Validation and Model Evaluation
Evaluate model performance using cross-validation and metrics.

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
```

#### Hyperparameter Tuning
Optimize model performance using grid search.

```python
from pyspark.ml.tuning import ParamGridBuilder

grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
```

## Prerequisites
- Basic knowledge of Python
- Familiarity with SQL is helpful but not required
- Understanding of fundamental machine learning concepts is beneficial

## Tools & Libraries
- Apache Spark
- PySpark
- Pandas
- SQL

## Learning Outcomes
By completing this course, you will:
- Understand the fundamentals of Apache Spark and PySpark
- Efficiently manipulate large datasets using Spark DataFrames
- Build and deploy machine learning pipelines with PySpark
- Optimize and evaluate machine learning models

---

Start your journey with PySpark and unlock the power of big data processing and machine learning!
Code explain the concept


# PySpark Example Project

This project demonstrates how to use PySpark for data processing and machine learning in Google Colab. It covers various functionalities such as data manipulation, SQL queries, feature engineering, and logistic regression classification.

---

## **Installation**
Before running the script, ensure that PySpark is installed. If you are using Google Colab, install PySpark with:

```python
!pip install pyspark
```

---

## **Code Overview**

### **1. Importing Necessary Libraries**
The script imports essential PySpark modules for handling data frames, SQL queries, and machine learning pipelines:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

### **2. Creating a Spark Session**
A Spark session is initialized to enable PySpark operations:

```python
spark = SparkSession.builder.appName("PySparkExample").getOrCreate()
```

### **3. Creating and Displaying a Sample DataFrame**
A sample dataset containing names and ages is created and displayed:

```python
data = [("Alice", 29), ("Bob", 31), ("Catherine", 25)]
df = spark.createDataFrame(data, ["Name", "Age"])
df.show()
```

### **4. Exploring the Data**
The Spark context version is printed, and the schema of the DataFrame is displayed:

```python
sc = spark.sparkContext
print("Spark Version:", sc.version)
df.printSchema()
```

### **5. SQL Query on DataFrame**
A temporary SQL table is created, and an SQL query is executed:

```python
df.createOrReplaceTempView("people")
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()
```

### **6. Converting DataFrame to Pandas**
The PySpark DataFrame is converted to a Pandas DataFrame:

```python
pandas_df = df.toPandas()
print(pandas_df.head())
```

### **7. Data Transformations**
- Adding a new column (`AgePlusOne`).
- Filtering rows where age is greater than 25.
- Selecting specific columns.
- Aggregating data by name.

```python
df = df.withColumn("AgePlusOne", col("Age") + 1)
df.filter(df.Age > 25).show()
df.select("Name").show()
df.groupBy("Name").agg(avg("Age")).show()
```

### **8. Feature Engineering**
A `StringIndexer` is used to convert categorical names into numerical values:

```python
indexer = StringIndexer(inputCol="Name", outputCol="NameIndex")
df = indexer.fit(df).transform(df)
df.show()
```

### **9. Splitting Data for Training and Testing**
The dataset is split into 80% training and 20% testing:

```python
train, test = df.randomSplit([0.8, 0.2])
```

### **10. Creating a Machine Learning Pipeline**
A `VectorAssembler` transforms the "Age" column into a feature vector:

```python
assembler = VectorAssembler(inputCols=["Age"], outputCol="features")
pipeline = Pipeline(stages=[assembler])
pipeline_model = pipeline.fit(train)
```

The transformations are applied to both training and test sets:

```python
train_transformed = pipeline_model.transform(train)
test_transformed = pipeline_model.transform(test)
```

### **11. Training a Logistic Regression Model**
A multinomial logistic regression model is trained on the transformed data:

```python
lr = LogisticRegression(featuresCol="features", labelCol="NameIndex", family="multinomial")
lr_model = lr.fit(train_transformed)
```

### **12. Evaluating the Model**
The model's accuracy is measured using `MulticlassClassificationEvaluator`:

```python
multi_evaluator = MulticlassClassificationEvaluator(labelCol="NameIndex", metricName="accuracy")
accuracy = multi_evaluator.evaluate(lr_model.transform(test_transformed))
print("Model Accuracy:", accuracy)
```

---

## **Expected Output**
- The dataset will be displayed after each transformation.
- The logistic regression model will train successfully.
- The accuracy of the model will be printed.

---

## **Conclusion**
This project demonstrates:
- How to create and manipulate DataFrames using PySpark.
- How to run SQL queries on DataFrames.
- How to perform feature engineering and train a machine learning model.
- How to evaluate a classification model in PySpark.

This serves as a basic foundation for working with PySpark in big data and machine learning applications!

---




```python
# Install PySpark in Google Colab
!pip install pyspark

# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder

# Create a Spark session
spark = SparkSession.builder.appName("PySparkExample").getOrCreate()

# Sample data
data = [("Alice", 29), ("Bob", 31), ("Catherine", 25)]
df = spark.createDataFrame(data, ["Name", "Age"])
df.show()

# Examine SparkContext
sc = spark.sparkContext
print("Spark Version:", sc.version)

# Print DataFrame Schema
df.printSchema()

# Create a temporary table and run an SQL query
df.createOrReplaceTempView("people")
sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()

# Convert to Pandas DataFrame
pandas_df = df.toPandas()
print(pandas_df.head())

# Add a new column
df = df.withColumn("AgePlusOne", col("Age") + 1)
df.show()

# Filter data
filtered_df = df.filter(df.Age > 25)
filtered_df.show()

# Select specific columns
df.select("Name").show()

# Aggregate data
df.groupBy("Name").agg(avg("Age")).show()

# Feature Engineering: Convert categorical data
indexer = StringIndexer(inputCol="Name", outputCol="NameIndex")
df = indexer.fit(df).transform(df)
df.show()

# Split data into training and test sets
train, test = df.randomSplit([0.8, 0.2])

# Machine Learning Pipeline
assembler = VectorAssembler(inputCols=["Age"], outputCol="features")
pipeline = Pipeline(stages=[assembler])
model = pipeline.fit(train)

# Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="NameIndex")
lr_model = lr.fit(train)

# Evaluate Model
evaluator = BinaryClassificationEvaluator(labelCol="NameIndex", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(lr_model.transform(test))
print("ROC-AUC Score:", roc_auc)

# Hyperparameter tuning
grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()

print("PySpark setup is complete and ready for data processing!")
```