import logging
import json
from pathlib import Path
from math import ceil, floor
from dataclasses import dataclass
import streamlit as st

# --- Translation System ---

@st.cache_data
def load_translations():
    """Load translation files for different languages."""
    translations = {}
    translation_path = Path(__file__).parent / "translations"
    
    # Try to load translations, fall back to English if not found
    for lang_code, lang_file in [("en", "en.json"), ("fr", "fr.json")]:
        try:
            with open(translation_path / lang_file, "r", encoding="utf-8") as f:
                translations[lang_code] = json.load(f)
        except FileNotFoundError:
            # If translation file not found, we'll handle it with a fallback
            pass
    
    # If no translations loaded, create minimal English fallback
    if "en" not in translations:
        translations["en"] = {
            "app_title": "LLM Inference GPU Calculator",
            "app_description": "Estimate the number of GPUs required to serve a Large Language Model (LLM)",
        }
    
    return translations

def get_text(key: str, **kwargs) -> str:
    """Get translated text for the current language."""
    lang = st.session_state.get("language", "en")
    translations = st.session_state.get("translations", {})
    
    # Get text from translations or fall back to key
    text = translations.get(lang, {}).get(key, translations.get("en", {}).get(key, key))
    
    # Format with any provided arguments
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    
    return text

# --- Data Models Configuration ---

@dataclass
class GpuConfig:
    """Represents the configuration of a GPU type."""
    display_name: str
    vram_gb: int
    flops_tflops: dict[str, float]
    meta: dict # For extra info like cost, sources, etc.

@dataclass
class ModelConfig:
    """Represents the configuration of a language model."""
    display_name: str
    size_b_params: float
    embed_dim: int
    n_layers: int
    meta: dict

# --- Constants ---
BYTES_PER_PARAM = {"8 bits (int8/fp8)": 1, "16 bits (fp16)": 2, "32 bits (fp32)": 4}

# GPU efficiency factors (real-world performance vs theoretical)
GPU_EFFICIENCY_FACTORS = {
    "8 bits (int8/fp8)": 0.45,    # ~45% efficiency in int8
    "16 bits (fp16)": 0.55,       # ~55% efficiency in fp16
    "32 bits (fp32)": 0.35,       # ~35% efficiency in fp32
}

# --- Configuration Loading ---

@st.cache_data
def load_app_config():
    """
    Loads GPU and Model configurations from JSON files in the 'db' directory.
    Uses dataclasses to structure the loaded data.
    """
    db_path = Path(__file__).parent / "db"
    config = {"gpus": {}, "models": {}}

    # Load GPUs
    try:
        with open(db_path / "gpu.json", "r") as f:
            gpu_data = json.load(f)
            for key, values in gpu_data.items():
                config["gpus"][key] = GpuConfig(**values)
    except FileNotFoundError:
        st.error(get_text("error_gpu_json_not_found"))
        return None
    except Exception as e:
        st.error(get_text("error_gpu_json_parse", error=str(e)))
        return None

    # Load Models
    try:
        with open(db_path / "models.json", "r") as f:
            model_data = json.load(f)
            for key, values in model_data.items():
                config["models"][key] = ModelConfig(**values)
    except FileNotFoundError:
        st.error(get_text("error_models_json_not_found"))
        return None
    except Exception as e:
        st.error(get_text("error_models_json_parse", error=str(e)))
        return None
        
    return config

# --- Model Configuration ---

def get_model_layers(model: ModelConfig) -> int:
    """
    Get the number of hidden layers for a model, 
    we estimate it from the model size if not provided.
    """
    return model.n_layers if model.n_layers else int(model.size_b_params**0.5 * 8)

# --- KV Cache Calculation ---

def calculate_kv_cache_memory(
    model: ModelConfig,
    max_batch_size: int,
    tokens_per_request: int,
    precision: str,
    concurrent_users: int
) -> float:
    """
    Calculates KV cache memory more accurately.
    
    Formula: 2 * num_layers * concurrent_users * seq_len * hidden_dim * bytes_per_param
    The 2 comes from K and V (Key and Value)
    
    Note: We use concurrent_users instead of batch_size because KV cache must be maintained
    for all active sequences, even when using batching strategies.
    """
    bytes_per_param = BYTES_PER_PARAM[precision]
    bytes_to_gb = 2**30
    
    model_layers = get_model_layers(model)
        
    kv_cache_bytes = (
        2 *  # K and V
        model_layers *
        model.embed_dim *
        concurrent_users *
        tokens_per_request *
        bytes_per_param
    )
    
    return kv_cache_bytes / bytes_to_gb

# --- Batching Strategies ---

def calculate_with_static_batching(total_rounds_needed: int, rounds_per_gpu: int) -> float:
    """Current strategy: fixed batch sizes"""
    return total_rounds_needed / rounds_per_gpu

def calculate_with_continuous_batching(total_rounds_needed: int, rounds_per_gpu: int) -> float:
    """
    Continuous batching: requests can join/leave dynamically
    Improves GPU utilization by ~1.8x on average
    """
    CONTINUOUS_BATCHING_EFFICIENCY = 1.8
    return total_rounds_needed / rounds_per_gpu / CONTINUOUS_BATCHING_EFFICIENCY

def calculate_with_dynamic_batching(total_rounds_needed: int, rounds_per_gpu: int) -> float:
    """
    Dynamic batching: adjusts batch size based on load
    More complex but can optimize latency vs throughput
    """
    DYNAMIC_BATCHING_EFFICIENCY = 1.3
    return total_rounds_needed / rounds_per_gpu / DYNAMIC_BATCHING_EFFICIENCY

# --- Prefill vs Decode ---

def calculate_effective_tokens(prompt_tokens: int, generated_tokens: int) -> float:
    """
    Prefill is ~10x faster per token than decode
    Returns effective token count for throughput calculations
    """
    PREFILL_SPEEDUP = 10
    return prompt_tokens / PREFILL_SPEEDUP + generated_tokens

# --- Calculation Logic Functions ---

def calculate_memory_constraint(
    model: ModelConfig,
    gpu: GpuConfig,
    precision: str,
    max_batch_size: int,
    tokens_per_request: int,
    memory_overhead_percent: float = 20,
    concurrent_users: int = 1,
) -> float:
    """
    Calculates the number of GPUs required to satisfy the VRAM memory constraint.

    Returns:
        The number of GPUs dictated by memory requirements.
    """
    bytes_per_param = BYTES_PER_PARAM[precision]
    bytes_to_gb = 2**30

    # Memory for the model weights
    model_mem_gb = model.size_b_params * bytes_per_param

    # Memory for the activation cache (approximate)
    # Formula: batch_size * seq_len * hidden_dim * bytes/param
    activations_mem_gb = (
        max_batch_size * tokens_per_request * model.embed_dim * bytes_per_param
    ) / bytes_to_gb

    # Memory for the KV Cache (more accurate calculation)
    kv_cache_mem_gb = calculate_kv_cache_memory(
        model, max_batch_size, tokens_per_request, precision, concurrent_users
    )

    # Apply memory overhead
    total_mem_gb = model_mem_gb + activations_mem_gb + kv_cache_mem_gb
    total_mem_gb *= (1 + memory_overhead_percent / 100)
    
    logging.info(
        f"Total memory required: {total_mem_gb:.2f} GB "
        f"(Model: {model_mem_gb:.2f}, Activations: {activations_mem_gb:.2f}, KV Cache: {kv_cache_mem_gb:.2f})"
    )
    
    return total_mem_gb / gpu.vram_gb


def calculate_speed_constraint(
    users: int,
    tps_per_user: int,
    max_batch_size: int,
    model: ModelConfig,
    gpu: GpuConfig,
    precision: str,
    batching_strategy: str = "Static Batching",
    effective_tokens_per_request: float = None,
) -> float:
    """
    Calculates the number of GPUs required to satisfy the speed (throughput) constraint.

    Returns:
        The number of GPUs dictated by speed, or infinity if impossible.
    """
    # Theoretical FLOPs to generate one token for this model (in TFLOPs)
    # Rule of thumb: 2 * N parameters (the '2' is an approximation for matrix operations)
    flops_per_token = model.size_b_params * 2 * 1e9 / 1e12 # from Giga-params to Tera-flops

    # FLOPs needed to generate one token for each sequence in the batch
    flops_per_batch_step = flops_per_token * max_batch_size

    # Apply GPU efficiency factor
    effective_gpu_flops = gpu.flops_tflops[precision] * GPU_EFFICIENCY_FACTORS[precision]
    
    # GPU throughput: how many "batch steps" (one token per sequence) can it perform per second?
    gpu_tps_per_sequence = effective_gpu_flops / flops_per_batch_step

    logging.info(f"Throughput per GPU: {gpu_tps_per_sequence:.2f} tokens/sec/sequence")
    
    # If the GPU's throughput is less than the required per-user throughput, it's impossible.
    if tps_per_user > gpu_tps_per_sequence:
        return float("inf")

    # How many "user groups" (of size tps_per_user) can be served on one GPU?
    rounds_per_gpu = floor(gpu_tps_per_sequence / tps_per_user)
    if not rounds_per_gpu:
        return float("inf")

    # Total number of batches to process to serve all users
    total_rounds_needed = ceil(users / max_batch_size)

    # Apply batching strategy
    if batching_strategy == "Static Batching":
        return calculate_with_static_batching(total_rounds_needed, rounds_per_gpu)
    elif batching_strategy == "Continuous Batching":
        return calculate_with_continuous_batching(total_rounds_needed, rounds_per_gpu)
    elif batching_strategy == "Dynamic Batching":
        return calculate_with_dynamic_batching(total_rounds_needed, rounds_per_gpu)
    else:
        return calculate_with_static_batching(total_rounds_needed, rounds_per_gpu)


def check_single_gpu_fit(
    model: ModelConfig,
    gpu: GpuConfig,
    precision: str,
    memory_overhead_percent: float = 20,
) -> tuple[bool, float]:
    """
    Check if the model can fit on a single GPU.
    Returns (can_fit, memory_used_gb)
    """
    bytes_per_param = BYTES_PER_PARAM[precision]
    model_mem_gb = model.size_b_params * bytes_per_param
    model_mem_gb *= (1 + memory_overhead_percent / 100)
    
    can_fit = model_mem_gb <= gpu.vram_gb
    return can_fit, model_mem_gb


def calculate_latency_metrics(
    tps_per_gpu: float,
    prompt_tokens: int,
    generated_tokens: int,
    batch_size: int,
) -> dict:
    """
    Calculate latency metrics (TTFT and token generation latency).
    """
    # Time to first token (prefill time)
    # Prefill is faster, we estimate 10x speedup
    # tps_per_gpu is tokens/second for the entire GPU
    # For a single sequence in the batch, divide by batch_size
    tps_per_sequence = tps_per_gpu / batch_size
    
    # Prefill processes all prompt tokens at once, with 10x speedup
    ttft = prompt_tokens / (tps_per_sequence * 10)
    
    # Time per output token during decode (no speedup, sequential generation)
    time_per_token = 1 / tps_per_sequence
    
    # Total generation time
    total_time = ttft + (generated_tokens * time_per_token)
    
    return {
        "ttft_ms": ttft * 1000,
        "time_per_token_ms": time_per_token * 1000,
        "total_time_s": total_time,
    }


# --- Streamlit User Interface ---

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(page_title="GpuCalculator", page_icon="💻")
    
    # Initialize session state for language if not set
    if "language" not in st.session_state:
        st.session_state.language = "en"
    
    # Load translations
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations()
    
    # Language selector in sidebar
    with st.sidebar:
        languages = {"en": "🇬🇧 English", "fr": "🇫🇷 Français"}
        selected_lang = st.selectbox(
            "Language / Langue",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            key="language",
        )

    # Database of available configurations
    APP_CONFIG = load_app_config()
    if not APP_CONFIG:
        st.error(get_text("error_config_load"))
        st.stop()
    
    st.title(get_text("app_title"))
    st.markdown(get_text("app_description"))
    st.markdown("---")

    # --- Section 1: Define Your Workload ---
    st.subheader(get_text("section_workload"))
    
    users = st.number_input(
        label=get_text("label_concurrent_users"),
        min_value=1,
        step=1,
        value=100,
        help=get_text("help_concurrent_users"),
    )
    
    tps_per_user = st.number_input(
        label=get_text("label_throughput_per_user"),
        min_value=1,
        step=1,
        value=10,
        help=get_text("help_throughput_per_user"),
    )
    
    # Optional latency requirements
    show_latency = st.checkbox(get_text("label_show_latency"), value=False)
    if show_latency:
        target_ttft = st.number_input(
            label=get_text("label_target_ttft"),
            min_value=10,
            value=200,
            help=get_text("help_target_ttft"),
        )

    # --- Section 2: Select Your Model ---
    st.subheader(get_text("section_model"))
    model_key = st.selectbox(
        label=get_text("label_choose_model"),
        options=list(APP_CONFIG["models"].keys()),
        format_func=lambda key: APP_CONFIG["models"][key].display_name,
        help=get_text("help_choose_model")
    )
    selected_model = APP_CONFIG["models"][model_key]
        
    precision = st.selectbox(
        label=get_text("label_quantization"),
        options=list(BYTES_PER_PARAM.keys()),
        help=get_text("help_quantization"),
    )
    
    tokens_per_req = st.number_input(
        label=get_text("label_context_size"),
        min_value=1,
        step=1,
        value=2048,
        help=get_text("help_context_size"),
    )
    
    # Prefill vs decode split
    with st.expander(get_text("label_advanced_prefill")):
        prompt_ratio = st.slider(
            get_text("label_prompt_ratio"),
            min_value=10,
            max_value=90,
            value=50,
            help=get_text("help_prompt_ratio"),
        )
        prompt_tokens = int(tokens_per_req * prompt_ratio / 100)
        generated_tokens = tokens_per_req - prompt_tokens
        st.caption(get_text("text_prompt_tokens", prompt_tokens=prompt_tokens, generated_tokens=generated_tokens))
        
        # Calculate effective tokens for throughput
        effective_tokens = calculate_effective_tokens(prompt_tokens, generated_tokens)
        st.caption(get_text("text_effective_tokens", effective_tokens=effective_tokens))

    # --- Section 3: Choose Hardware and Batching Strategy ---
    st.subheader(get_text("section_hardware"))
    
    gpu_key = st.selectbox(
        label=get_text("label_choose_gpu"),
        options=list(APP_CONFIG["gpus"].keys()),
        format_func=lambda key: APP_CONFIG["gpus"][key].display_name,
        help=get_text("help_choose_gpu")
    )
    selected_gpu = APP_CONFIG["gpus"][gpu_key]
    
    # Check if model fits on single GPU
    can_fit, model_mem = check_single_gpu_fit(
        selected_model, selected_gpu, precision
    )
    if not can_fit:
        st.error(
            get_text("warning_model_too_large", 
                    model_mem=model_mem, 
                    gpu_name=selected_gpu.display_name, 
                    gpu_vram=selected_gpu.vram_gb)
        )
    
    max_batch = st.number_input(
        label=get_text("label_batch_size"),
        min_value=1,
        step=1,
        value=8,
        help=get_text("help_batch_size"),
    )
    
    # Get batching strategy options
    batching_options = {
        "Static Batching": get_text("option_static_batching"),
        "Continuous Batching": get_text("option_continuous_batching"),
        "Dynamic Batching": get_text("option_dynamic_batching"),
    }
    
    batching_strategy = st.selectbox(
        label=get_text("label_batching_strategy"),
        options=list(batching_options.keys()),
        format_func=lambda x: batching_options[x],
        help=get_text("help_batching_strategy"),
    )

    # Memory overhead slider in advanced settings
    with st.expander(get_text("label_advanced_settings")):
        memory_overhead = st.slider(
            get_text("label_memory_overhead"), 
            min_value=0, 
            max_value=50, 
            value=20,
            help=get_text("help_memory_overhead")
        )

    st.markdown("---")

    # --- Calculation and Results Display ---
    if st.button(get_text("button_calculate"), type="primary", use_container_width=True):
        
        # Use effective tokens if prefill/decode split is considered
        effective_tokens_value = None
        if 'effective_tokens' in locals():
            effective_tokens_value = effective_tokens
        
        mem_constraint = calculate_memory_constraint(
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
            max_batch_size=max_batch,
            tokens_per_request=tokens_per_req,
            memory_overhead_percent=memory_overhead,
            concurrent_users=users,
        )

        speed_constraint = calculate_speed_constraint(
            users=users,
            tps_per_user=tps_per_user,
            max_batch_size=max_batch,
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
            batching_strategy=batching_strategy,
            effective_tokens_per_request=effective_tokens_value,
        )
        
        if speed_constraint == float("inf"):
            st.error(get_text("error_impossible_config"))
        else:
            required_gpus = max(mem_constraint, speed_constraint)
            
            st.success(get_text("success_gpus_required", gpu_count=ceil(required_gpus)))
            
            # Show efficiency warnings
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
            
            # Show latency metrics if requested
            if show_latency and 'prompt_tokens' in locals():
                # Calculate approximate GPU throughput
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
                    st.warning(
                        get_text("warning_ttft_exceeded", 
                                actual_ttft=latency_metrics['ttft_ms'], 
                                target_ttft=target_ttft)
                    )


if __name__ == "__main__":
    main()