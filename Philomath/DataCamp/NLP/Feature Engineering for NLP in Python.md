# Feature Engineering for NLP in Python

## Introduction
Feature engineering is a crucial step in Natural Language Processing (NLP). It involves extracting useful information from text and transforming it into a structured format suitable for machine learning models. In this guide, we will cover several fundamental NLP feature engineering techniques, including:

- **Basic text features and readability scores**
- **Text preprocessing, POS tagging, and Named Entity Recognition (NER)**
- **N-Gram models**
- **TF-IDF and similarity scores**

Each section includes explanations, mathematical formulas, and Python code with comments to illustrate how these techniques work.

---

## 1. Basic Features and Readability Scores
### Concept Explanation
Basic text features provide essential insights into the structure of a text document. These features help in text classification, sentiment analysis, and even readability estimation. 

Some common basic text features include:
- **Number of words**: Indicates the length of the text, useful for filtering short or long documents.
- **Number of characters**: Helps in understanding the text's density.
- **Average word length**: Affects readability; shorter words are easier to understand.
- **Number of special characters**: Special characters like `@` and `#` are common in social media and can be used for entity recognition.
- **Readability scores**: These scores (e.g., Flesch-Kincaid) assess the complexity of a text, indicating how difficult it is to read.

### Formulas:
- **Average Word Length** = \( \frac{\text{Total Characters}}{\text{Total Words}} \)
- **Flesch Reading Ease Score**:
  \[
  206.835 - (1.015 \times \text{ASL}) - (84.6 \times \text{ASW})
  \]
  where:
  - ASL = Average Sentence Length (words per sentence)
  - ASW = Average Syllables per Word

### Example Code:
```python
import textstat
import re

def basic_text_features(text):
    words = text.split()
    num_words = len(words)  # Count number of words
    num_chars = sum(len(word) for word in words)  # Count total characters
    avg_word_length = num_chars / num_words if num_words > 0 else 0  # Compute average word length
    num_special_chars = len(re.findall(r'[@#]', text))  # Count special characters (@, #)
    
    return {
        "num_words": num_words,
        "num_chars": num_chars,
        "avg_word_length": avg_word_length,
        "num_special_chars": num_special_chars
    }

text = "This is an example text with #hashtag and @mention."
print(basic_text_features(text))

# Compute readability score
readability_score = textstat.flesch_reading_ease(text)
print("Readability Score:", readability_score)
```

---

## 2. Text Preprocessing, POS Tagging, and Named Entity Recognition (NER)
### Concept Explanation
Text preprocessing is a fundamental step in NLP that prepares raw text for analysis. It includes:
- **Tokenization**: Splitting text into words or sentences.
- **Lemmatization**: Converting words to their base forms (e.g., "running" → "run").
- **Part-of-Speech (POS) Tagging**: Identifying grammatical categories (e.g., noun, verb, adjective) for each word.
- **Named Entity Recognition (NER)**: Identifying proper nouns, organizations, dates, etc.

These preprocessing steps help in building robust NLP models by making text more structured and interpretable.

### Example Code:
```python
import spacy

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Tokenization
    lemmas = [token.lemma_ for token in doc]  # Lemmatization
    pos_tags = [(token.text, token.pos_) for token in doc]  # POS tagging
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # Named Entity Recognition (NER)
    
    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities
    }

text = "Apple is looking at buying a U.K. startup for $1 billion."
print(process_text(text))
```

---

## 3. N-Gram Models
### Concept Explanation
An **N-Gram** is a sequence of `N` consecutive words used to analyze text patterns. 
- **Unigrams (N=1)**: Single words, useful for basic keyword analysis.
- **Bigrams (N=2)**: Two-word sequences, helpful for capturing relationships.
- **Trigrams (N=3)**: Three-word sequences, useful in language modeling.

### Formula:
For a sentence \( S \) consisting of words \( w_1, w_2, ..., w_n \), an N-gram sequence is:
\[ N\text{-gram} = (w_i, w_{i+1}, ..., w_{i+N-1}) \]

### Example Code:
```python
from sklearn.feature_extraction.text import CountVectorizer

text_data = ["This is a sample sentence", "Feature engineering for NLP"]
vectorizer = CountVectorizer(ngram_range=(2, 2))  # Generate bigrams (n=2)
X = vectorizer.fit_transform(text_data)

print("N-Gram Vocabulary:", vectorizer.get_feature_names_out())
print("N-Gram Features:", X.toarray())
```

---

## 4. TF-IDF and Similarity Scores
### Concept Explanation
TF-IDF (**Term Frequency - Inverse Document Frequency**) is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus). It helps remove common but less meaningful words (e.g., "the", "is").

### Formula:
- **TF (Term Frequency)**: \( TF(w) = \frac{\text{Count of word w in document}}{\text{Total words in document}} \)
- **IDF (Inverse Document Frequency)**: \( IDF(w) = \log\frac{\text{Total Documents}}{1 + \text{Number of documents containing w}} \)
- **TF-IDF**: \( TF-IDF(w) = TF(w) \times IDF(w) \)

### Example Code:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text_documents = [
    "NLP is an exciting field of study.",
    "Machine learning and NLP go hand in hand.",
    "Deep learning is useful in NLP."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_documents)

print("TF-IDF Matrix:")
print(X.toarray())

# Compute cosine similarity between documents
cosine_sim = cosine_similarity(X)
print("Cosine Similarity Matrix:")
print(cosine_sim)
```

---

## Conclusion
Feature engineering is an essential step in building NLP models. These techniques lay the foundation for advanced NLP tasks like **sentiment analysis, text classification, and recommendation systems**.

