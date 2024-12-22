"""
SCoRe Model - Self-Correcting Language Model with Reinforcement Learning
"""

from .config import Config
from .model import AdvancedModel
from .dataset import BaseDataset
from .trainer import SCoReTrainer
from .utils import set_seed, load_json, load_model

__version__ = "0.1.0"
__all__ = [
    'Config',
    'AdvancedModel',
    'BaseDataset',
    'SCoReTrainer',
    'set_seed',
    'load_json',
    'load_model'
] 