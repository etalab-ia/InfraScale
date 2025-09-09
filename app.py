import streamlit as st
from pathlib import Path
import json
from dataclasses import dataclass
from math import ceil
import logging

logging.basicConfig(filename='infrascale.log', level=logging.INFO)
logger = logging.getLogger("infrascale")

from language import load_translations, get_text, LANGUAGES
from utils import BYTES_PER_PARAM, GPU_EFFICIENCY_FACTORS
from solver import InfrascaleSolver
from constraints import calculate_speed_constants


@dataclass
class GpuConfig:
    display_name: str
    vram_gb: int
    flops_tflops: dict[str, float]
    bandwidth_tbps: float
    meta: dict

@dataclass
class ModelConfig:
    display_name: str
    size_b_params: float
    embed_dim: int
    n_layers: int
    meta: dict

def get_model_layers(model):
    return model.n_layers if model.n_layers else int(model.size_b_params**0.5 * 8)

@st.cache_data
def load_app_config():
    db_path = Path(__file__).parent / "db"
    config = {"gpus": {}, "models": {}}

    with open(db_path / "gpu.json", "r") as f:
        gpu_data = json.load(f)
        for key, values in gpu_data.items():
            config["gpus"][key] = GpuConfig(**values)

    with open(db_path / "models.json", "r") as f:
        model_data = json.load(f)
        for key, values in model_data.items():
            config["models"][key] = ModelConfig(**values)

    return config

def check_single_gpu_fit(model, gpu, precision, memory_overhead_percent=20):
    bytes_per_param = BYTES_PER_PARAM[precision]
    model_mem_gb = model.size_b_params * bytes_per_param
    model_mem_gb *= (1 + memory_overhead_percent / 100)
    return model_mem_gb <= gpu.vram_gb, model_mem_gb

def calculate_effective_tokens(prompt_tokens, generated_tokens):
    return prompt_tokens / 10 + generated_tokens

def get_min_ttft(**kwargs):
    if len(kwargs) == 0:
        return 2
    kwargs['max_wait'] = 1000
    return calculate_speed_constants(7, **kwargs)[1]

def main():
    st.set_page_config(page_title="Infrascale", page_icon="💻")
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations()

    with st.sidebar:
        st.selectbox("Language / Langue", list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x], key="language")

    APP_CONFIG = load_app_config()
    if not APP_CONFIG:
        st.error(get_text("error_config_load"))
        st.stop()

    st.title(get_text("app_title"))
    st.markdown(get_text("app_description"))
    st.markdown("---")

    st.subheader(get_text("section_workload"))
    st.markdown(get_text("text_workload_explanation"))
    users = st.number_input(get_text("label_concurrent_users"), min_value=1, value=100)
    tps_per_user = st.number_input(get_text("label_throughput_per_user"), min_value=1, value=10)

    params = {} if "params" not in st.session_state else st.session_state.params

    target_ttft = st.number_input(get_text("label_target_ttft"), min_value=1, value=5)

    st.subheader(get_text("section_model"))
    model_key = st.selectbox(
        get_text("label_choose_model"),
        options=list(APP_CONFIG["models"].keys()),
        format_func=lambda k: APP_CONFIG["models"][k].display_name
    )
    selected_model = APP_CONFIG["models"][model_key]

    precision = st.selectbox(get_text("label_quantization"), list(BYTES_PER_PARAM.keys()))

    tokens_per_req = st.number_input(get_text("label_context_size"), min_value=1, value=2048)

    with st.expander(get_text("label_advanced_prefill")):
        prompt_ratio = st.slider(get_text("label_prompt_ratio"), min_value=10, max_value=90, value=50)
        prompt_tokens = int(tokens_per_req * prompt_ratio / 100)
        generated_tokens = tokens_per_req - prompt_tokens
        st.caption(get_text("text_prompt_tokens", prompt_tokens=prompt_tokens, generated_tokens=generated_tokens))
        effective_tokens = calculate_effective_tokens(prompt_tokens, generated_tokens)
        st.caption(get_text("text_effective_tokens", effective_tokens=effective_tokens))

    st.subheader(get_text("section_hardware"))
    gpu_key = st.selectbox(
        get_text("label_choose_gpu"),
        options=list(APP_CONFIG["gpus"].keys()),
        format_func=lambda k: APP_CONFIG["gpus"][k].display_name,
        key="H100_80gb"
    )
    selected_gpu = APP_CONFIG["gpus"][gpu_key]

    can_fit, model_mem = check_single_gpu_fit(selected_model, selected_gpu, precision)
    if not can_fit:
        st.error(get_text("warning_model_too_large", model_mem=model_mem, gpu_name=selected_gpu.display_name, gpu_vram=selected_gpu.vram_gb))

    with st.expander(get_text("label_advanced_settings")):
        memory_overhead = st.slider(get_text("label_memory_overhead"), min_value=0, max_value=50, value=20)

    st.markdown("---")

    params = {
        'users': users,
        'target_speed': tps_per_user,
        'model_b_params': selected_model.size_b_params,
        'model_layers': selected_model.n_layers,
        'model_dim': selected_model.embed_dim,
        'bytes_per_param': BYTES_PER_PARAM[precision],
        'tokens_per_request': tokens_per_req,
        'memory_overhead_percent': memory_overhead,
        'prompt_size': prompt_tokens,
        'efficiency_factor': 0.85,
        'gpu_vram_gb': selected_gpu.vram_gb,
        'gpu_flops': selected_gpu.flops_tflops[precision],
        'gpu_bandwidth': selected_gpu.bandwidth_tbps,
        'gpu_efficiency_factor'	: GPU_EFFICIENCY_FACTORS[precision],
        'max_wait': target_ttft
    }

    solver = InfrascaleSolver(logger, **params)

    min_ttft = get_min_ttft(**params)
    if target_ttft < min_ttft:
        st.warning(get_text("warning_target_ttft_too_small", target_ttft=target_ttft))
        st.stop()

    if st.button(get_text("button_calculate"), type="primary", use_container_width=True):

        results = solver.solve()
        cluster_size, n_clusters, batch_size, queue_size = results.x
        # metrics = solver.get_metrics(cluster_size, n_clusters)

        required_gpus = ceil(cluster_size) * ceil(n_clusters)
        st.success(get_text("success_gpus_required", gpu_count=required_gpus))

        st.write(get_text("text_calculation_based_on"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(get_text("info_cluster_size", constraint=ceil(cluster_size)))
        with col2:
            st.info(get_text("info_n_clusters", constraint=ceil(n_clusters)))
        with col3:
            st.info(get_text("info_batch_size", constraint=2**batch_size))
            
        # if show_latency:
        #     st.markdown(get_text("heading_latency_estimates"))
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         st.metric(get_text("metric_ttft"), get_text("units_ms", value=metrics['tpot']))
        #     with col2:
        #         st.metric(get_text("metric_time_per_token"), get_text("units_ms", value=metrics['tpot']))
        #     with col3:
        #         st.metric(get_text("metric_total_generation"), get_text("units_seconds", value=metrics['tpot'] * effective_tokens))

        #     if metrics['tpot'] > target_ttft:
        #         st.warning(get_text("warning_ttft_exceeded", actual_ttft=metrics['tpot'], target_ttft=target_ttft))

if __name__ == "__main__":
    main()