import torch

def _rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)

def rotary_embed(
    q: torch.Tensor, k: torch.Tensor,
    positions: torch.Tensor, inv_freq: torch.Tensor,
) -> dict:
    """
    Returns q_rotated and k_rotated tensors in a dictionary.
    """
    if positions.ndim == 1:
        pos = positions[None, None, :, None]
    else:
        pos = positions[:, None, :, None]
    theta = pos * inv_freq
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    q_r = _rotary(q, sin, cos)
    k_r = _rotary(k, sin, cos)
    return {
        "q_rotated": q_r,
        "k_rotated": k_r,
    }
