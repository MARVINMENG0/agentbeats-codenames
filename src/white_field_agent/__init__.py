# White Agent Package
# This module exports the public API for the white agent (player).
# 
# The __init__.py file serves as the package's entry point and declares
# what is publicly accessible from this package.

from .server import start

# __all__ defines the public API - when someone does "from white import *",
# only the items listed here will be imported. This keeps internal implementation
# details (like Executor, agent.py, player.py) hidden and provides a clean interface.
__all__ = ["start"]