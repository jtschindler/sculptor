from importlib.metadata import PackageNotFoundError, version

from sculptor.logging_config import configure_logging, reset_logging

try:
    __version__ = version("sculptor")
except PackageNotFoundError:
    # Not installed (e.g. imported straight from a source checkout). The
    # version itself is set statically in pyproject.toml.
    __version__ = "0.0.0"

# Show sculptor's INFO messages by default (see sculptor.logging_config).
configure_logging()

__all__ = ["__version__", "configure_logging", "reset_logging"]
