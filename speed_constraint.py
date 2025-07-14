import logging
from math import ceil, floor
from utils import GPU_EFFICIENCY_FACTORS, BYTES_PER_PARAM

def calculate_with_static_batching(total_rounds_needed, rounds_per_gpu):
    return total_rounds_needed / rounds_per_gpu

def calculate_with_continuous_batching(total_rounds_needed, rounds_per_gpu):
    return total_rounds_needed / rounds_per_gpu / 1.8

def calculate_with_dynamic_batching(total_rounds_needed, rounds_per_gpu):
    return total_rounds_needed / rounds_per_gpu / 1.3

def calculate_effective_tokens(prompt_tokens, generated_tokens):
    return prompt_tokens / 10 + generated_tokens

def calculate_speed_constraint(users, tps_per_user, max_batch_size, model, gpu, precision, batching_strategy="Static Batching", effective_tokens_per_request=None):
    flops_per_token = model.size_b_params * 2 * 1e9 / 1e12
    flops_per_batch_step = flops_per_token * max_batch_size
    effective_gpu_flops = gpu.flops_tflops[precision] * GPU_EFFICIENCY_FACTORS[precision]

    gpu_tps_per_sequence = effective_gpu_flops / flops_per_batch_step
    logging.info(f"Throughput per GPU: {gpu_tps_per_sequence:.2f} tokens/sec/sequence")

    if tps_per_user > gpu_tps_per_sequence:
        return float("inf")

    rounds_per_gpu = floor(gpu_tps_per_sequence / tps_per_user)
    if not rounds_per_gpu:
        return float("inf")

    total_rounds_needed = ceil(users / max_batch_size)

    strategy_fn = {
        "Static Batching": calculate_with_static_batching,
        "Continuous Batching": calculate_with_continuous_batching,
        "Dynamic Batching": calculate_with_dynamic_batching,
    }.get(batching_strategy, calculate_with_static_batching)

    return strategy_fn(total_rounds_needed, rounds_per_gpu)

def calculate_latency_metrics(tps_per_gpu, prompt_tokens, generated_tokens, batch_size):
    tps_per_sequence = tps_per_gpu / batch_size
    ttft = prompt_tokens / (tps_per_sequence * 10)
    time_per_token = 1 / tps_per_sequence
    total_time = ttft + (generated_tokens * time_per_token)

    return {
        "ttft_ms": ttft * 1000,
        "time_per_token_ms": time_per_token * 1000,
        "total_time_s": total_time,
    }
