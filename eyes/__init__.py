"""Eyes monitoring system package."""
# Import subpackages
from . import config  # This is the config package
from . import core
from . import gui
from .config import *  # Import all configuration constants

__all__ = ['config', 'core', 'gui']