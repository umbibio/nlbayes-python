"""nlbayes - Bayesian Networks for TF Activity Inference"""

from .ornor import ORNOR
from .ModelORNOR import PyModelORNOR as ModelORNOR  # Keep for backward compatibility

__all__ = ["ModelORNOR", "ORNOR"]

__version__ = "0.8.1"
