import logging
from math import ceil
from utils import BYTES_PER_PARAM, get_model_layers

def calculate_kv_cache_memory(model, max_batch_size, tokens_per_request, precision, concurrent_users):
    bytes_per_param = BYTES_PER_PARAM[precision]
    bytes_to_gb = 2**30
    model_layers = get_model_layers(model)

    kv_cache_bytes = (
        2 * model_layers * model.embed_dim * max_batch_size *
        tokens_per_request * bytes_per_param
    )
    return kv_cache_bytes / bytes_to_gb

def calculate_memory_constraint(model, gpu, precision, max_batch_size, tokens_per_request, memory_overhead_percent=20, concurrent_users=1):
    bytes_per_param = BYTES_PER_PARAM[precision]
    bytes_to_gb = 2**30

    model_mem_gb = model.size_b_params * bytes_per_param
    activations_mem_gb = (max_batch_size * tokens_per_request * model.embed_dim * bytes_per_param) / bytes_to_gb
    kv_cache_mem_gb = calculate_kv_cache_memory(model, max_batch_size, tokens_per_request, precision, concurrent_users)

    total_mem_gb = (model_mem_gb + activations_mem_gb + kv_cache_mem_gb) * (1 + memory_overhead_percent / 100)

    logging.info(f"Total memory required: {total_mem_gb:.2f} GB (Model: {model_mem_gb:.2f}, Activations: {activations_mem_gb:.2f}, KV Cache: {kv_cache_mem_gb:.2f})")
    return total_mem_gb / gpu.vram_gb
