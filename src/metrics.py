import torch
from dataclasses import dataclass

@dataclass
class TensorStats:
    name: str
    mean: float
    std: float
    L1: float
    L2: float
    min: float
    max: float
    norm: float

def compute_tensor_stats(tensor: torch.Tensor, name: str) -> TensorStats:
    with torch.no_grad():
        tensor = tensor.float().view(-1)
        mean = tensor.mean().item()
        std = tensor.std().item()
        L1 = tensor.abs().mean().item()
        L2 = tensor.pow(2).mean().sqrt().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        norm = torch.norm(tensor).item()

        return TensorStats(name, mean, std, L1, L2, min_val, max_val, norm)