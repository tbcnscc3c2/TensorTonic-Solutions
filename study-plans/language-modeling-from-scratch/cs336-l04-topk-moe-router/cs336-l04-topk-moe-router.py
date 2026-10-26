import torch

def topk_moe_router(x: torch.Tensor, w_router: torch.Tensor, expert_outputs: torch.Tensor, shared_output: torch.Tensor, k: int) -> dict:
    """
    Returns tensors in a dict: output, selected_experts, selected_weights, token_fraction, probability_mass.
    """
    scores = x.float() @ w_router.float()
    T, E = scores.shape
    selected_experts = torch.argsort(
        scores, dim=-1, descending=True, stable=True
    )[..., :k]
    selected_scores = scores.gather(dim=-1, index=selected_experts)
    selected_expert_outputs = expert_outputs.gather(dim=1, index=selected_experts.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1]))
    selected_expert_outputs = selected_expert_outputs.float()
    alpha = torch.softmax(selected_scores, dim=-1)
    y = (alpha.unsqueeze(-1) * selected_expert_outputs).sum(dim=1) + shared_output
    counts = torch.bincount(
        selected_experts.reshape(-1),
        minlength=E
    )
    f_e = counts / T
    m_e = torch.softmax(scores, dim=-1).mean(dim=0)
    return {
        "output": y.to(x.dtype),
        "selected_experts": selected_experts,
        "selected_weights": alpha.to(x.dtype),
        "token_fraction": f_e.to(x.dtype),
        "probability_mass": m_e.to(x.dtype)
    }

