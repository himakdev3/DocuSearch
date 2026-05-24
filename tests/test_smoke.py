from src.config.config_manager import config
from src.utils.runtime_env import configure_runtime_environment


def test_config_validation_shape() -> None:
    issues = config.validate()
    assert isinstance(issues, list)


def test_runtime_environment_sets_expected_vars() -> None:
    configure_runtime_environment()

    import os

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
