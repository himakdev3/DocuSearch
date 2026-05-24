"""Runtime environment bootstrap utilities.

These variables must be configured before importing heavy ML libraries so
threading and tokenizer behavior are deterministic across platforms.
"""

import os

TOKENIZERS_PARALLELISM = "false"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"


def configure_runtime_environment() -> None:
    """Apply safe runtime defaults for tokenizers and numerical backends."""
    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = MKL_NUM_THREADS
