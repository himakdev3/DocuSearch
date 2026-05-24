"""Runtime environment bootstrap utilities.

These variables must be configured before importing heavy ML libraries so
threading and tokenizer behavior are deterministic across platforms.
"""

import os

TOKENIZERS_PARALLELISM = "false"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
STREAMLIT_SERVER_FILE_WATCHER_TYPE = "none"


def configure_runtime_environment() -> None:
    """Apply safe runtime defaults for tokenizers and numerical backends."""
    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS
    # Prevent Streamlit from probing every transformers submodule, which can
    # trigger optional torchvision imports on environments without torchvision.
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = STREAMLIT_SERVER_FILE_WATCHER_TYPE
