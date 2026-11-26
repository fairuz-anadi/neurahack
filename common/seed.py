import os
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def device_info() -> dict:
    info = {"cpu_count": os.cpu_count(), "gpu": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info
