import random

import numpy as np
import torch


def set_seed(seed):
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
