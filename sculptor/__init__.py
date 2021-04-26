#!/usr/bin/env python

import pkg_resources

# Set the global version variable from the setup.py file
try:
    __version__ = pkg_resources.get_distribution('sculptor').version
except Exception:
    __version__ = 'unknown'