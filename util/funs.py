import random

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed):
    """Fix every RNG we use so a run is reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
