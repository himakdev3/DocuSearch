
# Must be set before torch / tokenizers are imported to prevent segfault on macOS.
from src.config.config_manager import config
from src.utils.logging_config import configure_logging
from src.utils.runtime_env import configure_runtime_environment


def main() -> None:
    """Application entrypoint for Streamlit."""
    configure_logging(config.log_level, config.log_format, config.logs_dir)
    configure_runtime_environment()

    # Import after runtime env configuration so library thread settings are honored.
    from src.interfaces.streamlit_app import run_streamlit_app

    run_streamlit_app()


if __name__ == "__main__":
    main()



