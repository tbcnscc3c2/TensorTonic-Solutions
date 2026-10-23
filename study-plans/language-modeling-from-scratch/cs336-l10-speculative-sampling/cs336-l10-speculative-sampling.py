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
    q = draft_probabilities[positions, draft_tokens].clamp(min=0.0)
    p = target_probabilities[positions, draft_tokens].clamp(min=0.0)
    a = (p / q).nan_to_num(nan=0).clamp(max=1.0)
    reject = (a <= acceptance_random_values)
    reject_pos = torch.where(reject == True)[0]
    if reject_pos.numel() == 0:
        return {
            "tokens": draft_tokens,
            "accepted_count": draft_tokens.numel(),
            "rejected_at": None
        }
    reject_pos = reject_pos[0].item()
    r_u = (target_probabilities[reject_pos] - draft_probabilities[reject_pos]).clamp(min=0.0)
    r = r_u / r_u.sum()
    cdf = torch.cumsum(r, dim=-1)
    replace_token = torch.searchsorted(cdf, residual_random_value, right=True).unsqueeze(0)
    if replace_token.item() == r.numel():
        positive_indices = torch.nonzero(r > 0, as_tuple=True)[0]
        replace_token = positive_indices[-1:]
    return {
        "tokens": torch.cat((draft_tokens[:reject_pos], replace_token)),
        "accepted_count": reject_pos,
        "rejected_at": reject_pos
    }
    
