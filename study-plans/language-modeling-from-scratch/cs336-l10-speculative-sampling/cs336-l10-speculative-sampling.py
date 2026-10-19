import torch

def speculative_sampling(
    draft_tokens: torch.Tensor, draft_probabilities: torch.Tensor,
    target_probabilities: torch.Tensor, acceptance_random_values: torch.Tensor,
    residual_random_value: int | float,
) -> dict:
    """
    Returns a dict: tokens (int64 tensor), accepted_count (int), rejected_at (int or None).
    """
    positions = torch.arange(draft_tokens.numel())
    p = target_probabilities[positions, draft_tokens]
    q = draft_probabilities[positions, draft_tokens]
    a = torch.nan_to_num(p / q, nan=0.0, posinf=1.0).clamp(max=1.0)

    accepted = a <= acceptance_random_values
    rejected_pos = torch.nonzero(accepted, as_tuple=True)[0]
    if rejected_pos.numel() == 0:
        return {
            "tokens": draft_tokens,
            "accepted_count": draft_tokens.numel(),
            "rejected_at": None
            }

    rejected_pos = rejected_pos[0].item()
    p = target_probabilities[rejected_pos]
    q = draft_probabilities[rejected_pos]
    weight = (p - q).clamp(min=0)
    r = weight / weight.sum()
    cdf = r.cumsum(dim=-1)
    replace_token = torch.searchsorted(cdf, residual_random_value, right=True)
    if replace_token.item() == r.numel():
        positive_indices = torch.nonzero(r > 0, as_tuple=True)[0]
        replace_token = positive_indices[-1]
    output_tokens = torch.cat([draft_tokens[:rejected_pos], replace_token.reshape(1)])
    return {
        "tokens": output_tokens,
        "accepted_count": rejected_pos,
        "rejected_at": rejected_pos
    }
    
