import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = inputs @ weights + bias
    y = torch.tanh(a)
    theta = upstream_gradient * (1 - y ** 2)
    x_grad = theta * weights
    w_grad = theta * inputs
    return (y, x_grad, w_grad, theta)
    