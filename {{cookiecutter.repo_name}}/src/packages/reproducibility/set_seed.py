import random

import numpy as np

SEED = 42


def seed_file(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across numpy and random modules."""
    random.seed(seed)
    np.random.seed(seed)
