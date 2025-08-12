from math import log10, ceil

class InfrascaleSolver:
    """Implementation of the Infrascale algorithm.
       We apply KKT theorem for different batch sizes and we keep the best solution.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    def _calculate_kv_cache_memory(self):
        kwargs = self.params

        model_layers = kwargs['model_layers']
        model_dim = kwargs['model_dim']
        batch_size = kwargs['batch_size']
        tokens_per_request = kwargs['tokens_per_request']
        bytes_per_param = kwargs['bytes_per_param']

        bytes_to_gb = 2**30

        kv_cache_bytes = (
            2 * model_layers * model_dim * batch_size *
            tokens_per_request * bytes_per_param
        )
        return kv_cache_bytes / bytes_to_gb

    def _calculate_memory_constant(self):
        kwargs = self.params
        
        model_b_params = kwargs['model_b_params']
        model_layers = kwargs['model_layers']
        model_dim = kwargs['model_dim']
        gpu_vram_gb = kwargs['gpu_vram_gb']
        batch_size = kwargs['batch_size']
        tokens_per_request = kwargs['tokens_per_request']
        bytes_per_param = kwargs['bytes_per_param']
        memory_overhead_percent = kwargs['memory_overhead_percent']

        bytes_to_gb = 2**30

        model_mem_gb = model_b_params * bytes_per_param
        activations_mem_gb = (batch_size * tokens_per_request * model_dim * bytes_per_param) / bytes_to_gb
        kv_cache_mem_gb = self._calculate_kv_cache_memory()

        total_mem_gb = (model_mem_gb + activations_mem_gb + kv_cache_mem_gb) * (1 + memory_overhead_percent / 100)

        return total_mem_gb / gpu_vram_gb

    def _calculate_speed_constant(self):
        kwargs = self.params

        users = kwargs['users']
        batch_size = kwargs['batch_size']
        prompt_size = kwargs['prompt_size']
        model_b_params = kwargs['model_b_params']
        model_dim = kwargs['model_dim']
        gpu_flops = kwargs['gpu_flops']
        gpu_bandwidth = kwargs['gpu_bandwidth']
        gpu_efficiency_factor = kwargs['gpu_efficiency_factor']
        
        gpu_flops = gpu_flops * 10**12 * gpu_efficiency_factor
        gpu_bandwidth = gpu_bandwidth * 10**12

        flops_decode_one_token = batch_size * model_b_params * 2 * 1e9
        mm_params = model_b_params*10**9
        mm_kv_read = 2 * batch_size * prompt_size * model_dim
        mm_kv_write = 2 * batch_size * model_dim
        mm_decode_one_token = 2 * (mm_params + mm_kv_read + mm_kv_write)
        tpot = max(flops_decode_one_token / gpu_flops, mm_decode_one_token / gpu_bandwidth)

        k = 24/model_b_params
        recalibration_factor = k**(-1/(k+1))
        calibrated_tpot = tpot / recalibration_factor

        throughput_one_gpu = 1 / calibrated_tpot

        return throughput_one_gpu

    def _calculate_S(self, N=1):
        target_speed = self.params['target_speed']
        users = self.params['users']
        batch_size = self.params['batch_size']
        E = self.params['efficiency_factor']
        K_s = self._calculate_speed_constant()
        K_m = self._calculate_memory_constant()

        S_1 = (K_m - 1) / E + 1
        S_2 = target_speed * log10(min(users/N, batch_size)+30) / (K_s * E) - 1 / E + 1

        return ceil(max(1, S_1, S_2))
    
    def _calculate_N(self):
        N = max(1, self.params['users'] / self.params['batch_size'])

        return ceil(N)
    
    def _calculate_GPU_needs_for_batch_size(self, batch_size):
        kwargs = self.params
        kwargs['batch_size'] = batch_size
        N = self._calculate_N()
        S = self._calculate_S(N)

        if S > self.params['batch_size']:
          raise ValueError("S is greater than batch_size")

        return ceil(S), ceil(N)

    def compute_GPU_needs(self):
        batchsize_range = [16, 32, 64, 128, 256, 512]
        first_S, first_N = self._calculate_GPU_needs_for_batch_size(batchsize_range[0])
        best_estimate = [first_S, first_N, batchsize_range[0]]
        for batch_size in batchsize_range[1:]:
            S, N = self._calculate_GPU_needs_for_batch_size(batch_size)
            if S * N < best_estimate[0] * best_estimate[1]:
                best_estimate[0] = S
                best_estimate[1] = N
                best_estimate[2] = batch_size

        return best_estimate
    
    def get_metrics(self, S, N):
        throughput_per_gpu = self._calculate_speed_constant()
        throughput_per_cluster = throughput_per_gpu * (1 + self.params['efficiency_factor'] * (S - 1))
        throughput_total = throughput_per_cluster * N
        throughput_per_user = throughput_total / self.params['users']
        tpot = 1 / throughput_per_cluster

        return {
            'throughput_per_gpu': throughput_per_gpu,
            'throughput_per_cluster': throughput_per_cluster,
            'throughput_total': throughput_total,
            'throughput_per_user': throughput_per_user,
            'tpot': tpot
        }