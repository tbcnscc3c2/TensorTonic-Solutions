import math

def kv_cache_capacity_planner(
    model_parameters: int, bytes_per_parameter: int, num_layers: int,
    sequence_length: int, num_attention_heads: int, num_kv_heads: int,
    head_dim: int, bytes_per_element: int, memory_per_gpu: int,
    reserved_memory: int, memory_bandwidth: int,
) -> dict:
    """
    Returns a dict: model_bytes, kv_bytes_per_request, max_batch_size, seconds_per_token, tokens_per_second.
    """
    m_model = model_parameters * bytes_per_parameter
    m_request = 2 * num_layers * sequence_length * num_kv_heads * head_dim * bytes_per_element
    B = math.floor((memory_per_gpu - reserved_memory - m_model) / m_request)
    B = max(0, B)
    T = (m_model + B * m_request) / memory_bandwidth
    return {
        "model_bytes": m_model,
        "kv_bytes_per_request": m_request,
        "max_batch_size": B,
        "seconds_per_token": T,
        "tokens_per_second": B / T
    }
