#Vectorizer Utility Class

#Overview
The VectorizerUtility repository provides a comprehensive utility class for text vectorization and document ranking. The utility class is designed to be compatible with scikit-learn's ColumnTransformer and includes custom implementations for BM25 and BM42 with additional functionalities for document ranking and similarity scoring.

#Features
  Vectorization Methods: BM25, and BM42.
  Custom BM25 and BM42 Classes: Implementations include methods for fitting, transforming, and ranking documents.
  Compatibility: Works seamlessly with scikit-learn's ColumnTransformer.
  Error Handling: Enhanced error handling to manage cases where the model needs to be fitted before transformation

#Implementation Notebook
The test.ipynb notebook demonstrates the usage of the VectorizerUtility class and its methods. It provides examples of vectorizing text data, ranking documents, and integrating with scikit-learn's ColumnTransformer.
