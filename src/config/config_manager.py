"""
Configuration settings for RAG Document Assistant

Production-ready configuration management with environment variables.
All settings can be overridden via .env file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from project .env file
try:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
    logger.info("Environment variables loaded from .env file")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}. Using defaults.")


def _get_env_bool(key: str, default: bool = False) -> bool:
    """
    Parse boolean from environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found or invalid
        
    Returns:
        Boolean value
    """
    try:
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    except Exception as e:
        logger.warning(f"Error parsing boolean for {key}: {e}. Using default: {default}")
        return default


def _get_env_int(key: str, default: int) -> int:
    """
    Parse integer from environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found or invalid
        
    Returns:
        Integer value
    """
    try:
        value = os.getenv(key)
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing integer for {key}: {e}. Using default: {default}")
        return default


def _get_env_float(key: str, default: float) -> float:
    """
    Parse float from environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not found or invalid
        
    Returns:
        Float value
    """
    try:
        value = os.getenv(key)
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing float for {key}: {e}. Using default: {default}")
        return default


class Config:
    """
    Centralized configuration loaded from environment variables.
    
    All settings default to production-ready values and can be overridden
    via .env file or environment variables.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        
        # ====================================================================
        # MODEL CONFIGURATION
        # ====================================================================
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_batch_size = _get_env_int("EMBEDDING_BATCH_SIZE", 16)
        self.use_gpu = _get_env_bool("USE_GPU", False)
        
        # LLM settings
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.llm_temperature = _get_env_float("LLM_TEMPERATURE", 0.7)
        self.llm_max_tokens = _get_env_int("LLM_MAX_TOKENS", 500)
        
        # ====================================================================
        # API KEYS
        # ====================================================================
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # ====================================================================
        # DOCUMENT PROCESSING
        # ====================================================================
        self.chunk_size = _get_env_int("CHUNK_SIZE", 500)
        self.chunk_overlap = _get_env_int("CHUNK_OVERLAP", 50)
        self.min_chunk_length = _get_env_int("MIN_CHUNK_LENGTH", 100)
        self.max_file_size_mb = _get_env_int("MAX_FILE_SIZE_MB", 50)
        
        # ====================================================================
        # SEARCH & RETRIEVAL
        # ====================================================================
        self.default_top_k = _get_env_int("DEFAULT_TOP_K", 5)
        self.max_top_k = _get_env_int("MAX_TOP_K", 15)
        self.min_similarity_threshold = _get_env_float("MIN_SIMILARITY_THRESHOLD", 0.3)
        
        # FAISS index settings
        self.use_gpu_index = _get_env_bool("USE_GPU_INDEX", False)
        self.index_type = os.getenv("INDEX_TYPE", "FlatIP")
        
        # ====================================================================
        # APPLICATION SETTINGS
        # ====================================================================
        self.app_title = os.getenv("APP_TITLE", "DocuSearch")
        self.app_icon = os.getenv("APP_ICON", "📚")
        self.page_layout = os.getenv("PAGE_LAYOUT", "wide")

        # ====================================================================
        # PATHS
        # ====================================================================
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.base_dir / "data" / "pdfs"
        self.logs_dir = self.base_dir / "output" / "logs"

        # Create necessary directories
        self._create_directories()

        # Session
        self.session_timeout_minutes = _get_env_int("SESSION_TIMEOUT_MINUTES", 60)
        self.enable_caching = _get_env_bool("ENABLE_CACHING", True)
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_summary(self) -> dict:
        """Get configuration summary as dictionary."""
        return {
            "model": {
                "embedding_model": self.embedding_model_name,
                "batch_size": self.embedding_batch_size,
                "use_gpu": self.use_gpu,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model
            },
            "document_processing": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "min_chunk_length": self.min_chunk_length,
                "max_file_size_mb": self.max_file_size_mb
            },
            "search": {
                "default_top_k": self.default_top_k,
                "max_top_k": self.max_top_k,
                "similarity_threshold": self.min_similarity_threshold,
                "index_type": self.index_type
            },
            "application": {
                "title": self.app_title,
                "icon": self.app_icon,
                "layout": self.page_layout,
                "caching_enabled": self.enable_caching
            }
        }
    
    def validate(self) -> list:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages (empty if all valid)
        """
        issues = []
        
        # Validate chunk settings
        if self.chunk_size < 100:
            issues.append("CHUNK_SIZE should be at least 100 characters")
        
        if self.chunk_overlap >= self.chunk_size:
            issues.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        # Validate search settings
        if self.default_top_k > self.max_top_k:
            issues.append("DEFAULT_TOP_K cannot exceed MAX_TOP_K")
        
        if not 0 <= self.min_similarity_threshold <= 1:
            issues.append("MIN_SIMILARITY_THRESHOLD must be between 0 and 1")
        
        # Warn about API keys if LLM features are used
        if self.llm_provider == "openai" and not self.openai_api_key:
            issues.append("WARNING: OPENAI_API_KEY not set (required for LLM features)")
        
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            issues.append("WARNING: ANTHROPIC_API_KEY not set (required for LLM features)")
        
        return issues
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config(model={self.embedding_model_name}, chunk_size={self.chunk_size}, top_k={self.default_top_k})"


# Global configuration instance
config = Config()
