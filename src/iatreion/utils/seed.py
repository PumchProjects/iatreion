import os
import random
from collections.abc import Sequence

import numpy as np


def set_device(device_id: int | str | Sequence[int]) -> None:
    if isinstance(device_id, str):
        value = device_id
    elif isinstance(device_id, Sequence):
        value = ','.join(str(item) for item in device_id)
    else:
        value = str(device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_seed_torch(seed: int) -> None:
    import torch

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
