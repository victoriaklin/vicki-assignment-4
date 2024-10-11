import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Transform the documents into a TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(documents)

# Apply SVD for dimensionality reduction
svd = TruncatedSVD(n_components=100)  # Adjust the number of components as needed
X_reduced = svd.fit_transform(X)

# Function to process a query and retrieve top 5 similar documents
def retrieve_documents(query, top_n=5):
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)
    similarities = cosine_similarity(query_reduced, X_reduced)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return [(i, documents[i], similarities[0, i]) for i in top_indices]