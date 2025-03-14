# Natural Language Processing (NLP) - ReadMe

## 1. Regular Expressions & Word Tokenization
**Summary:** This chapter introduces basic NLP concepts like word tokenization and regular expressions to help parse text. It also covers handling non-English text and challenging tokenization cases.

### Example (Python):
```python
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

text = "Hello, world! Welcome to NLP."
tokens = word_tokenize(text)
print("Word Tokens:", tokens)

# Using Regular Expressions
pattern = r'\b\w+\b'
words = re.findall(pattern, text)
print("Regex Tokens:", words)
```

---

## 2. Simple Topic Identification
**Summary:** This chapter introduces topic identification using basic NLP models. Topics are identified from texts based on term frequencies, using methods like bag-of-words and TF-IDF with NLTK and Gensim.

### Example (Python):
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["NLP is amazing and fun.", "Machine learning makes NLP better.", "Deep learning enhances NLP."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:")
print(X.toarray())
```

---

## 3. Named-Entity Recognition (NER)
**Summary:** This chapter covers Named Entity Recognition (NER), which identifies entities like names, locations, and dates using pre-trained models like spaCy and Polyglot.

### Example (Python):
```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Elon Musk founded SpaceX in 2002."
doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
```

---

## 4. Building a "Fake News" Classifier
**Summary:** This chapter applies supervised machine learning techniques to build a "fake news" detector by selecting important features and testing ideas for classifying fake news articles.

### Example (Python):
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample data
text_data = ["Breaking news: AI will take over the world!", "Scientists discover water on Mars.", "Fake news: The moon is made of cheese."]
labels = [1, 0, 1]  # 1 = Fake, 0 = Real

# Create a model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(text_data, labels)

# Predict new input
prediction = model.predict(["NASA announces new Mars mission."])
print("Prediction (0 = Real, 1 = Fake):", prediction)
```

---
This README provides an overview and simple Python implementations for each NLP concept.

