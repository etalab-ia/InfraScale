import logging
from math import ceil, floor, log
from utils import GPU_EFFICIENCY_FACTORS, BYTES_PER_PARAM

logger = logging.getLogger("infrascale")

def calculate_with_static_batching(throughput):
    return throughput

def calculate_with_continuous_batching(throughput):
    return throughput / 1.8

def calculate_with_dynamic_batching(throughput):
    return throughput / 1.3

def calculate_effective_tokens(prompt_tokens, generated_tokens):
    return prompt_tokens / 10 + generated_tokens

def calculate_calibrated_tpot(time_decode_one_token, users, batch_size, model_params):
    k = 24e9/model_params
    recalibration_factor = k**(-1/(k+1))
    return time_decode_one_token / recalibration_factor * log(min(users, batch_size)+30,10)

def calculate_speed_constraint(users, target_throughput, max_batch_size, prompt_size, output_size, model, gpu, precision, batching_strategy="Static Batching", effective_tokens_per_request=None, target_ttft=1, weights_constraint=1):
    # for a given batch, we need to compute:
    # - the time to generate the first token (prefill), linked to flops and bandwidth
    # - the time to generate the rest of the tokens (decode), linked to flops and bandwidth
    # and then deduce 1) the resulting throughput, assuming 100 outputs tokens per request
    # 2) the number of sequential rounds that hold in the GPU while still maintining a throughput greater than expected tps_per_user
    cluster_efficiency = 1 + 0.85 * (weights_constraint - 1)
    gpu_flops = gpu.flops_tflops[precision] * GPU_EFFICIENCY_FACTORS[precision] #todo
    gpu_flops = gpu_flops * 10**12 * cluster_efficiency
    gpu_bandwidth = gpu.bandwidth_tbps 
    gpu_bandwidth = gpu_bandwidth * 10**12 * cluster_efficiency
    generated_tokens = output_size
    #prompt_size = prompt_size

    #something looks wrong in the modelization of prefill and decode, maybe the formula is outdated. TODO
    flops_prefill = 2 * max_batch_size * model.n_layers * prompt_size * model.embed_dim * (2*model.embed_dim + prompt_size) #TODO
    mm_prefill = BYTES_PER_PARAM[precision] * (model.size_b_params * 1e9 + 2 * model.n_layers * max_batch_size * prompt_size * model.embed_dim + max_batch_size * prompt_size * model.embed_dim) #TODO
    time_prefill = max(flops_prefill / gpu_flops, mm_prefill / gpu_bandwidth)

    #this formula is the classical approximation of flops for a decode step, but it looks outdated too. TODO
    flops_decode_one_token = max_batch_size * model.size_b_params * 2 * 1e9 #TODO
    mm_params = model.size_b_params*10**9
    mm_kv_read = 2 * max_batch_size * prompt_size * model.embed_dim
    mm_kv_write = 2 * max_batch_size * model.embed_dim
    mm_decode_one_token = BYTES_PER_PARAM[precision] * (mm_params + mm_kv_read + mm_kv_write) #TODO
    time_decode_one_token = max(flops_decode_one_token / gpu_flops, mm_decode_one_token / gpu_bandwidth)

    tpot = calculate_calibrated_tpot(time_decode_one_token, users, max_batch_size, model.size_b_params)

    throughput = 1 / tpot
    
    # ttft_constraint = time_prefill / target_ttft #TODO

    logger.info(f"Throughput per GPU cluster : {throughput:.2f} tokens/sec")

    if throughput < target_throughput:
        return float("inf")

    strategy_fn = {
        "Static Batching": calculate_with_static_batching,
        "Continuous Batching": calculate_with_continuous_batching,
        "Dynamic Batching": calculate_with_dynamic_batching,
    }.get(batching_strategy, calculate_with_static_batching)

    throughput_constraint = target_throughput / strategy_fn(throughput)

    return throughput_constraint * weights_constraint

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
