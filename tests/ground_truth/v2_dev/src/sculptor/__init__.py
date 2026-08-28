#!/usr/bin/env python

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sculptor")
except PackageNotFoundError:
    __version__ = "unknown"