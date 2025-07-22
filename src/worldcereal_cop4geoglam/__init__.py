#!/usr/bin/env python3
"""WorldCereal Copernicus4GEOGLAM package."""

import sys

from loguru import logger

from ._version import __version__

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO"
)

# Make logger and version available
__all__ = ["logger", "__version__"]
