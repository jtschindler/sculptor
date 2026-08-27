try:
    from importlib.metadata import version
    __version__ = version("sculptor-extensions")
except BaseException:
    __version__ = "unknown"