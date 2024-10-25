from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping, get_psuedo_label_weights, get_terminal_weights


def fit(data, network, **kwargs):
    """
    Fit Feature Extractor, Task Classifier by DeepJDOT algo.
    https://arxiv.org/abs/1803.10081
    """
    pass