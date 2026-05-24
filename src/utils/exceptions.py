"""
Custom exceptions for RAG Document Assistant.

This module defines custom exception classes for better error handling
throughout the application.
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors."""
    pass


class ConfigurationError(RAGException):
    """Raised when there's an error in configuration."""
    pass


class ModelLoadError(RAGException):
    """Raised when embedding model fails to load."""
    pass


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass


class PDFExtractionError(DocumentProcessingError):
    """Raised when PDF text extraction fails."""
    pass


class EmbeddingGenerationError(RAGException):
    """Raised when embedding generation fails."""
    pass


class IndexBuildError(RAGException):
    """Raised when FAISS index build fails."""
    pass


class SearchError(RAGException):
    """Raised when search/retrieval fails."""
    pass


class ValidationError(RAGException):
    """Raised when validation fails."""
    pass
