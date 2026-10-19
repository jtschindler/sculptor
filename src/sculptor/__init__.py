from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sculptor")
except PackageNotFoundError:
    # Not installed (e.g. imported straight from a source checkout). The
    # version itself is set statically in pyproject.toml.
    __version__ = "0.0.0"
