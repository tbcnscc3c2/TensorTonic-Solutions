def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    weights = [weight.clone() for weight in weights]
    biases = [bias.clone() for bias in biases]

    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)  # [bs] → [bs, 1]

    losses = []
    with torch.no_grad():
        for step in range(steps+1):
            activations = [inputs]
            for weight, bias in zip(weights, biases):
                h = torch.tanh(activations[-1] @ weight.T + bias)
                activations.append(h)
            loss = torch.pow(activations[-1] - targets, 2).sum()

            if step == steps:
                break

            losses.append(loss)
            grad_h =  2 * (activations[-1] - targets)
            for idx in range(len(weights) - 1, -1, -1):
                h_prev = activations[idx]
                h = activations[idx + 1]
                weight = weights[idx]
                bias = biases[idx]

                delta = grad_h * (1 - h ** 2)
                grad_w = delta.T @ h_prev
                grad_b =  delta.sum(dim=0)
                grad_h = delta @ weight

                weight -= learning_rate * grad_w
                bias -= learning_rate * grad_b                
                
        return (activations[-1].squeeze(-1), loss, weights, biases, losses)