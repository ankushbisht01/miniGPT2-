import torch
import random
import numpy as np

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device(preferred: str = "mps") -> torch.device:
    # Resolution priority: mps → cuda → cpu
    if torch.backends.mps.is_available() and preferred == "mps":
        return torch.device("mps")
    if torch.cuda.is_available() and preferred == "gpu":
        return torch.device("gpu")
    return torch.device("cpu")

    