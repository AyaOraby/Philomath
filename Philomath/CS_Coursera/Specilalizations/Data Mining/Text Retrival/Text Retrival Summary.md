# Text Retrieval (TR) Concepts

## 1. NLP as the Foundation for Text Retrieval

- **Definition**: Natural Language Processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language.
- NLP techniques, such as tokenization, stemming, lemmatization, and named entity recognition, help improve text search quality and relevance.
- While advanced NLP models enhance semantic understanding, basic Bag-of-Words (BoW) models—where text is treated as a set of words without considering order or meaning—are often sufficient for ranking and retrieval tasks.
- The trade-off between computational efficiency and retrieval accuracy determines the choice of NLP models in real-world search applications.

### Key NLP Techniques in Text Retrieval

#### 1. Tokenization

- **Definition**: Splitting text into words, phrases, or symbols to facilitate processing.
- **Example Code (Python - NLTK & spaCy)**:

```python
import nltk
from nltk.tokenize import word_tokenize
import spacy

nltk.download('punkt')
text = "Text retrieval is an important part of search engines."
tokens_nltk = word_tokenize(text)

nlp = spacy.load("en_core_web_sm")
tokens_spacy = [token.text for token in nlp(text)]

print("NLTK Tokenization:", tokens_nltk)
print("spaCy Tokenization:", tokens_spacy)
```

#### 2. Stemming

- **Definition**: Reducing words to their root form by removing suffixes to standardize terms.
- **Example Code:**

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["running", "retrieved", "computational", "retrieval"]
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words:", stemmed_words)
```

#### 3. Lemmatization

- **Definition**: Converts words to their base dictionary form while preserving meaning.
- **Example Code:**

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
words = ["running", "retrieved", "better", "retrieval"]
lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
print("Lemmatized Words:", lemmatized_words)
```

#### 4. Named Entity Recognition (NER)

- **Definition**: Identifies proper nouns like names, locations, and dates in text.
- **Example Code:**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Google was founded by Larry Page and Sergey Brin in 1998 in California."
doc = nlp(text)
print("Named Entities:", [(ent.text, ent.label_) for ent in doc.ents])
```

## 2. Push vs. Pull; Task-Based vs. Browsing

- **Push Systems**: Information is proactively recommended based on user behavior, preferences, or contextual signals (e.g., news feed recommendations, personalized ads).
- **Pull Systems**: Users actively search for specific information using queries in a search engine (e.g., Google, academic databases).
- **Task-Based Search**: Users have a well-defined goal (e.g., searching for a specific document, troubleshooting a technical issue).
- **Browsing**: Users explore information without a specific goal in mind (e.g., scrolling through an e-commerce site, discovering new articles).
- **Definition**: Understanding user intent in these paradigms helps optimize search experience, personalization, and retrieval efficiency.

### Code Example for Push & Pull Systems

```python
class SearchSystem:
    def __init__(self, mode):
        self.mode = mode

    def search(self, query=None):
        if self.mode == "push":
            return "Recommended articles based on preferences."
        elif self.mode == "pull":
            return f"Results for query: {query}"
        else:
            return "Invalid mode"

pull_search = SearchSystem("pull")
print(pull_search.search("Machine Learning"))

push_search = SearchSystem("push")
print(push_search.search())
```

## 3. Text Retrieval as a Ranking Problem

- **Definition**: Text retrieval involves ranking documents based on relevance to a user query.
- Documents are ranked based on scoring functions that measure relevance using lexical, syntactic, and semantic features.
- **Traditional ranking methods** include TF-IDF (Term Frequency-Inverse Document Frequency), BM25, and cosine similarity.
- **Learning-to-Rank (LTR) models** leverage machine learning to optimize ranking based on labeled training data.
- The quality of ranking directly affects user satisfaction and information discovery.

### Code Example for TF-IDF Ranking

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["Text retrieval is important", "Ranking improves search results", "Machine learning enhances ranking"]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
```

# Combination of Retrieval Methods

## 4. Combination of Retrieval Methods

### Definition
Combining multiple retrieval methods, such as keyword-based, semantic, and neural retrieval, enhances search accuracy by leveraging the strengths of each approach.

### Hybrid Search Models
Hybrid search models integrate traditional retrieval techniques (e.g., TF-IDF, BM25) with deep learning-based models (e.g., dense vector embeddings from BERT) to improve the quality and relevance of search results.

#### Example Implementation in Python
Below is an example of a hybrid search implementation using `TF-IDF` and `BERT-based embeddings` with `Faiss` for efficient similarity search.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample documents
documents = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing enables computers to understand and respond to human language.",
]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# BERT-based Embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = bert_model.encode(documents)

# FAISS Index for Fast Nearest Neighbor Search
d = bert_embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(bert_embeddings).astype('float32'))

# Querying
def hybrid_search(query, alpha=0.5):
    # Compute TF-IDF and BERT embedding for query
    tfidf_query = tfidf_vectorizer.transform([query])
    bert_query = bert_model.encode([query])[0]
    
    # FAISS Search for nearest neighbor
    D, I = index.search(np.array([bert_query]).astype('float32'), k=1)
    
    # Hybrid scoring
    scores = alpha * D[0] + (1 - alpha) * tfidf_query.dot(tfidf_matrix.T).toarray()[0]
    
    # Retrieve best match
    best_match = np.argmax(scores)
    return documents[best_match]

query = "How does AI understand text?"
result = hybrid_search(query)
print("Best Match:", result)
```

### Explanation
- **TF-IDF** captures keyword-based relevance.
- **BERT embeddings** provide semantic understanding.
- **Faiss** efficiently retrieves the most similar document.
- The **hybrid search function** combines both techniques using a weighted score.

This approach improves search accuracy by leveraging both lexical and contextual similarities.

## 5. Inverted Index for Efficient Search

### Definition
An index structure that maps words to document locations, enabling fast lookups.

### Used in
Search engines, databases, and large-scale text retrieval systems.

## 6. Evaluation Metrics for Information Retrieval

### Definition
Methods to assess retrieval performance.

### Common Metrics
- Precision
- Recall
- F1-score
- Mean Average Precision (MAP)
- NDCG (Normalized Discounted Cumulative Gain)

## 7. Feedback Mechanisms for Search Improvement

### Definition
Systems that refine search results based on user interaction.

### Examples
- Relevance feedback
- Click-through rate analysis
- Query expansion

## 8. Parallel Indexing Using MapReduce

### Definition
A distributed computing framework for processing large-scale indexing tasks.

### Usage
Improves scalability and efficiency of search engine indexing.

## 9. Future of Web Search

### Advancements in AI
- Integration of deep learning, reinforcement learning, and large-scale neural networks.

### Voice and Multimodal Search
- Expanding beyond text to image, audio, and video search.

### Personalization and Privacy
- Balancing customization with user data protection.

