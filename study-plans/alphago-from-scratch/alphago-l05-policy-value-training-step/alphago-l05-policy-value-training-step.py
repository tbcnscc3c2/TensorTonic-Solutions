import torch

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    policy_weight, policy_bias, value_weight, value_bias = [
        param.detach().clone().float().requires_grad_(True)
        for param in (policy_weight, policy_bias, value_weight, value_bias)
    ]
    p_a = features @ policy_weight.T + policy_bias
    v = torch.tanh(features @ value_weight.T + value_bias)
    p_loss = -(target_policies * torch.log_softmax(p_a, dim=-1)).sum(dim=-1).mean()
    v_loss = torch.square(v - target_values).mean()
    r_loss = l2_coefficient * (
        policy_weight.square().sum()
        + policy_bias.square().sum()
        + value_weight.square().sum()
        + value_bias.square().sum()
    )
    loss = p_loss + v_loss + r_loss
    loss.backward()
    p_weight = (policy_weight - learning_rate * policy_weight.grad).detach().clone()
    p_bias = (policy_bias - learning_rate * policy_bias.grad).detach().clone()
    v_weight = (value_weight - learning_rate * value_weight.grad).detach().clone()
    v_bias = (value_bias - learning_rate * value_bias.grad).detach().clone()
    return (loss.detach().clone(), p_weight, p_bias, v_weight, v_bias)
