import streamlit as st
from pathlib import Path
import json
from dataclasses import dataclass
from math import ceil
import logging

logging.basicConfig(filename='infrascale.log', level=logging.INFO)
logger = logging.getLogger("infrascale")

from language import load_translations, get_text
from memory_constraint import calculate_memory_constraint, BYTES_PER_PARAM
from speed_constraint import (
    calculate_speed_constraint,
    calculate_latency_metrics,
    calculate_effective_tokens,
    GPU_EFFICIENCY_FACTORS
)


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

def main():
    st.set_page_config(page_title="GpuCalculator", page_icon="💻")
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations()

    with st.sidebar:
        languages = {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}
        st.selectbox("Language / Langue", list(languages.keys()), format_func=lambda x: languages[x], key="language")

    APP_CONFIG = load_app_config()
    if not APP_CONFIG:
        st.error(get_text("error_config_load"))
        st.stop()

    st.title(get_text("app_title"))
    st.markdown(get_text("app_description"))
    st.markdown("---")

    st.subheader(get_text("section_workload"))
    users = st.number_input(get_text("label_concurrent_users"), min_value=1, value=100)
    tps_per_user = st.number_input(get_text("label_throughput_per_user"), min_value=1, value=10)

    show_latency = st.checkbox(get_text("label_show_latency"), value=False)
    if show_latency:
        target_ttft = st.number_input(get_text("label_target_ttft"), min_value=10, value=200)

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
        format_func=lambda k: APP_CONFIG["gpus"][k].display_name
    )
    selected_gpu = APP_CONFIG["gpus"][gpu_key]

    can_fit, model_mem = check_single_gpu_fit(selected_model, selected_gpu, precision)
    if not can_fit:
        st.error(get_text("warning_model_too_large", model_mem=model_mem, gpu_name=selected_gpu.display_name, gpu_vram=selected_gpu.vram_gb))

    max_batch = st.number_input(get_text("label_batch_size"), min_value=1, value=8)

    batching_options = {
        "Static Batching": get_text("option_static_batching"),
        "Continuous Batching": get_text("option_continuous_batching"),
        "Dynamic Batching": get_text("option_dynamic_batching"),
    }
    batching_strategy = st.selectbox(
        get_text("label_batching_strategy"),
        list(batching_options.keys()),
        format_func=lambda x: batching_options[x]
    )

    with st.expander(get_text("label_advanced_settings")):
        memory_overhead = st.slider(get_text("label_memory_overhead"), min_value=0, max_value=50, value=20)

    st.markdown("---")

    if st.button(get_text("button_calculate"), type="primary", use_container_width=True):

        speed_constraint = calculate_speed_constraint(
            users=users,
            target_throughput=tps_per_user,
            max_batch_size=max_batch,
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
            batching_strategy=batching_strategy,
            effective_tokens_per_request=effective_tokens,
        )

        mem_constraint = calculate_memory_constraint(
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
            max_batch_size=max_batch,
            tokens_per_request=tokens_per_req,
            memory_overhead_percent=memory_overhead,
            concurrent_users=users,
            speed_constraint=speed_constraint,
        )

        if speed_constraint == float("inf"):
            st.error(get_text("error_impossible_config"))
        else:
            required_gpus = max(mem_constraint, speed_constraint)
            st.success(get_text("success_gpus_required", gpu_count=ceil(required_gpus)))

            if required_gpus < 1 and batching_strategy == "Static Batching":
                st.info(get_text("tip_underutilized"))
            if max_batch > 32:
                st.warning(get_text("warning_large_batch"))

            st.write(get_text("text_calculation_based_on"))
            col1, col2 = st.columns(2)
            with col1:
                st.info(get_text("info_memory_constraint", constraint=mem_constraint))
            with col2:
                st.info(get_text("info_speed_constraint", constraint=speed_constraint))

            if mem_constraint > speed_constraint:
                st.warning(get_text("warning_memory_bottleneck"))
            else:
                st.warning(get_text("warning_compute_bottleneck"))

            if show_latency:
                effective_gpu_flops = selected_gpu.flops_tflops[precision] * GPU_EFFICIENCY_FACTORS[precision]
                flops_per_token = selected_model.size_b_params * 2 * 1e9 / 1e12
                gpu_tps = effective_gpu_flops / flops_per_token

                latency_metrics = calculate_latency_metrics(
                    gpu_tps, prompt_tokens, generated_tokens, max_batch
                )

                st.markdown(get_text("heading_latency_estimates"))
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_text("metric_ttft"), get_text("units_ms", value=latency_metrics['ttft_ms']))
                with col2:
                    st.metric(get_text("metric_time_per_token"), get_text("units_ms", value=latency_metrics['time_per_token_ms']))
                with col3:
                    st.metric(get_text("metric_total_generation"), get_text("units_seconds", value=latency_metrics['total_time_s']))

                if latency_metrics['ttft_ms'] > target_ttft:
                    st.warning(get_text("warning_ttft_exceeded", actual_ttft=latency_metrics['ttft_ms'], target_ttft=target_ttft))

if __name__ == "__main__":
    main()