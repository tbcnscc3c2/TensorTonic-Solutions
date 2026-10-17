import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = inputs @ weights + bias
    y = torch.tanh(a)
    delta = upstream_gradient * (1 - y ** 2)
    inputs_grad = delta * weights
    weights_grad = delta * inputs
    return (y, inputs_grad, weights_grad, delta)