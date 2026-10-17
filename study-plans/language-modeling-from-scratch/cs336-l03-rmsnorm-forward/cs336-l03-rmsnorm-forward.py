import torch

def rmsnorm(x: torch.Tensor, g: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Returns the RMS-normalized tensor with the same shape and dtype as x.
    """
    rms_inv = torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + epsilon)
    return x * rms_inv * g
