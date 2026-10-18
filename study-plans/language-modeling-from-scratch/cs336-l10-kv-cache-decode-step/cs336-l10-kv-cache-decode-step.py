import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    B, Hq, D = query.shape
    assert num_query_heads % num_kv_heads == 0
    key_cache = torch.cat([key_cache, new_key.unsqueeze(-2)], dim=-2)
    value_cache = torch.cat([value_cache, new_value.unsqueeze(-2)], dim=-2)

    G = num_query_heads // num_kv_heads
    key = key_cache.repeat_interleave(G, dim=1)
    value = value_cache.repeat_interleave(G, dim=1)
    score = torch.softmax(query.unsqueeze(-2) @ key.transpose(-2, -1) * torch.rsqrt(torch.tensor(D)), dim=-1)
    output = score @ value
    return {
        "output": output.squeeze(-2),
        "new_key_cache": key_cache,
        "new_value_cache": value_cache
    }
