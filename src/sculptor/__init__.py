from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sculptor")
except PackageNotFoundError:
    # Not yet installed -- e.g. `pyproject.toml`'s
    # `[tool.setuptools.dynamic] version = {attr = "sculptor.__version__"}`
    # imports this module during the package's own build, before its
    # metadata is registered. "unknown" is not a valid PEP 440 version and
    # would make that build fail, so fall back to a valid placeholder.
    __version__ = "0.0.0"
