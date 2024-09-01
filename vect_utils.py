from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy.linalg import norm
from numpy import dot


# Custom transformer for MultiLabelBinarizer
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

# Custom transformer for BM25
"""
BM25 Explanation:

BM25 is a probabilistic information retrieval model used to rank documents based on the frequency of query terms within them. 
It’s an extension of the TF-IDF model and incorporates document length normalization and term frequency saturation.

BM25 Scoring Formula:
---------------------
The BM25 score for a term 't' in a document 'd' is calculated as:

    BM25(d, t) = IDF(t) * [(f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * (|d| / avgdl)))]

Where:
    - f(t, d): Term frequency of 't' in document 'd'.
    - |d|: The length of document 'd'.
    - avgdl: The average document length across the entire corpus.
    - k1: A tunable parameter that controls term frequency scaling (default is 1.5).
    - b: A tunable parameter that controls document length normalization (default is 0.75).
    - IDF(t): Inverse Document Frequency of term 't', which is calculated as:
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5)) + 1
        Where:
            - N: Total number of documents.
            - df(t): Document frequency of term 't' (i.e., the number of documents containing 't').

Key Points:
-----------
- BM25 accounts for both term frequency (within the document) and document length.
- It is more flexible than TF-IDF due to the tunable parameters 'k1' and 'b'.
- BM25 penalizes the overuse of frequent terms in documents, making it effective for ranking.

--------------------------------------------------------------------------------

"""
class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1=1.5, b=0.75, max_features=None, ngram_range=(1, 1)):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.CountVectorizerBM = None
        self.doc_freqs = None
        self.idf = None
        self.avgdl = None

    def fit(self, X, y=None):
        # Initialize the vectorizer
        self.CountVectorizerBM = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        X_count = self.CountVectorizerBM.fit_transform(X)
        tokenized_corpus = X_count.toarray()
        
        # Store document frequencies and other parameters
        self.doc_freqs = np.sum((tokenized_corpus > 0), axis=0)
        doc_lengths = np.array([len(doc.split()) for doc in X])
        self.avgdl = np.mean(doc_lengths)
        self.idf = np.log((len(X) - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5)) + 1
        
        return self

    def transform(self, X):
        if not self.CountVectorizerBM:
            raise ValueError("The model must be fitted before transforming.")
        
        X_count = self.CountVectorizerBM.transform(X)
        tokenized_corpus = X_count.toarray()
        
        doc_lengths = np.array([len(doc.split()) for doc in X])
        
        bm_vectors = []
        for i in range(len(X)):
            bm_vector = []
            for j in range(len(self.CountVectorizerBM.get_feature_names_out())):
                tf = tokenized_corpus[i][j]
                numerator = self.idf[j] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_lengths[i] / self.avgdl))
                bm_score = numerator / denominator
                bm_vector.append(bm_score)
            bm_vectors.append(bm_vector)

        return np.array(bm_vectors)

    def rank_documents(self, query, documents):
        if not self.CountVectorizerBM:
            # Model is not fitted, so fit it with the provided documents
            print("Model not fitted. Fitting the model with provided documents...")
            self.fit(documents)

        query_vector = self.transform([query])[0]
        document_vectors = self.transform(documents)

        p1 = query_vector.reshape(1,-1).dot(document_vectors.T)
        p2 = norm(document_vectors.T,axis=0)*norm(query_vector.reshape(1,-1))
        scores = p1/p2
        return np.argsort(-scores), scores

# Custom transformer for BM42
"""
BM42 Explanation:

BM42 is a variation of BM25 that introduces an additional parameter, k2, to account for the term frequency within the query itself. 
This allows for more nuanced control over how query terms influence the document ranking.

BM42 Scoring Formula:
---------------------
The BM42 score for a document 'd' given a query 'q' is calculated as:

    BM42(d, q) = sum( IDF(t) * [(f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * (|d| / avgdl)))] * [(f(t, q) * (k2 + 1)) / (f(t, q) + k2)] )

Where:
    - f(t, d): Term frequency of 't' in document 'd'.
    - f(t, q): Term frequency of 't' in the query 'q'.
    - |d|: The length of document 'd'.
    - avgdl: The average document length across the entire corpus.
    - k1: A tunable parameter for term frequency scaling in documents (default is 1.5).
    - k2: A tunable parameter for term frequency scaling in queries (default is 1.0).
    - b: A tunable parameter that controls document length normalization (default is 0.75).
    - IDF(t): Inverse Document Frequency of term 't', calculated as in BM25.

Key Points:
-----------
- BM42 extends BM25 by incorporating the term frequency within the query itself, controlled by the k2 parameter.
- This is particularly useful for handling queries with repeated terms or for more complex information retrieval tasks.
- BM42 provides finer control over the ranking process by allowing adjustments to both document and query term frequencies.

--------------------------------------------------------------------------------
"""
class BM42Transformer(BaseEstimator, TransformerMixin):
        def __init__(self, k1=1.5, b=0.75, k2=1.0, max_features=None, ngram_range=(1, 1)):
            self.k1 = k1
            self.b = b
            self.k2 = k2
            self.max_features = max_features
            self.ngram_range = ngram_range
            self.CountVectorizerBM = None
            self.doc_freqs = None
            self.idf = None
            self.avgdl = None

        def fit(self, X, y=None):
            """
            Fit the BM42 model on the training data.

            :param documents: List of training documents.
            :param max_features: Maximum number of features (terms) to consider.
            :param ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
            :param k1: BM k1 parameter (controls term frequency scaling).
            :param b: BM b parameter (controls document length normalization).
            :param k2: BM k2 parameter (additional term frequency scaling, only for BM42).
            """
            # Initialize the vectorizer
            self.CountVectorizerBM = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
            X_count = self.CountVectorizerBM.fit_transform(X)
            tokenized_corpus = X_count.toarray()
            
            # Store document frequencies and other parameters
            self.doc_freqs = np.sum((tokenized_corpus > 0), axis=0)
            doc_lengths = np.array([len(doc.split()) for doc in X])
            self.avgdl = np.mean(doc_lengths)
            self.idf = np.log((len(X) - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5)) + 1

            return self

        def transform(self, X):
            """
            Transform new documents using the fitted BM25 or BM42 model.

            :param documents: List of documents to transform.
            :return: BM42 vectorized representation of documents.
            """
            if not self.CountVectorizerBM:
                raise ValueError("The model must be fitted before transforming.")

            X_count = self.CountVectorizerBM.transform(X)
            tokenized_corpus = X_count.toarray()
            
            doc_lengths = np.array([len(doc.split()) for doc in X])
            
            bm_vectors = []
            for i in range(len(X)):
                bm_vector = []
                for j in range(len(self.CountVectorizerBM.get_feature_names_out())):
                    tf = tokenized_corpus[i][j]
                    numerator = self.idf[j] * tf * (self.k1 + 1) * (self.k2 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_lengths[i] / self.avgdl)) * (tf + self.k2)
                    bm_score = numerator / denominator
                    bm_vector.append(bm_score)
                bm_vectors.append(bm_vector)

            return np.array(bm_vectors)
        
        def rank_documents(self, query, documents):
            if not self.CountVectorizerBM:
                # Model is not fitted, so fit it with the provided documents
                print("Model not fitted. Fitting the model with provided documents...")
                self.fit(documents)

            query_vector = self.transform([query])[0]
            document_vectors = self.transform(documents)

            # scores = []
            # for i, doc_vector in enumerate(document_vectors):
            #     score = 0
            #     for j in range(len(doc_vector)):
            #         tf_query = query_vector[j]
            #         numerator = doc_vector[j] * (self.k2 + 1)
            #         denominator = tf_query + self.k2
            #         score += numerator / denominator
            #     scores.append(score)
            
            p1 = (query_vector+self.k2).reshape(1,-1).dot((document_vectors*(self.k2 + 1)).T)
            p2 = norm((document_vectors*(self.k2 + 1)).T,axis=0)*norm((query_vector+self.k2).reshape(1,-1))
            scores = p1/p2

            return np.argsort(-scores), scores

#contains misc methods useful for vector modifications
class misc:
    def stack_vectors_horizontally(*vectors):
        """
        Stacks multiple arrays horizontally (column-wise).
        
        Parameters:
        *vectors : list of numpy arrays
            The vectors to stack horizontally.
        
        Returns:
        numpy.ndarray
            The stacked array.
        
        Raises:
        ValueError: If the arrays do not have the same number of rows.
        """
        # Ensure all vectors are numpy arrays
        arrays = [np.array(vec) for vec in vectors]

        # Check if all arrays have the same number of rows
        num_rows = arrays[0].shape[0]
        for array in arrays:
            if array.shape[0] != num_rows:
                raise ValueError("All arrays must have the same number of rows to stack horizontally.")

        # Stack them horizontally
        stacked_array = np.hstack(arrays)
        return stacked_array