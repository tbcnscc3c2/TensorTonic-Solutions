import torch
import math

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    batch, query_heads, sequence, head_width = q.shape
    batch, kv_heads, sequence, head_width = k.shape
    
    assert query_heads % kv_heads == 0

    group_num = query_heads // kv_heads
    q_group = q.reshape(batch, kv_heads, group_num, sequence, head_width)
    k_group = k.unsqueeze(2)
    v_group = v.unsqueeze(2)

    causal_mask = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
    scores = (q_group @ k_group.transpose(-2, -1)) / math.sqrt(head_width)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return (weights @ v_group).reshape(batch, query_heads, sequence, head_width)
