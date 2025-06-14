import logging
import json
from pathlib import Path
from math import ceil, floor
from dataclasses import dataclass
import streamlit as st

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
    meta: dict

# --- Constants ---
BYTES_PER_PARAM = {"8 bits (int8/fp8)": 1, "16 bits (fp16)": 2, "32 bits (fp32)": 4}

# --- Configuration Loading ---

@st.cache_data
def load_app_config():
    """
    Loads GPU and Model configurations from JSON files in the 'db' directory.
    Uses dataclasses to structure the loaded data.
    """
    #path is ./db/gpu.json and ./db/models.json
    db_path = Path(__file__).parent / "db"
    config = {"gpus": {}, "models": {}}

    # Load GPUs
    try:
        with open(db_path / "gpu.json", "r") as f:
            gpu_data = json.load(f)
            for key, values in gpu_data.items():
                config["gpus"][key] = GpuConfig(**values)
    except FileNotFoundError:
        st.error(f"Error: `db/gpu.json` not found. Please create it.")
        return None
    except Exception as e:
        st.error(f"Error parsing `db/gpu.json`: {e}")
        return None

    # Load Models
    try:
        with open(db_path / "models.json", "r") as f:
            model_data = json.load(f)
            for key, values in model_data.items():
                config["models"][key] = ModelConfig(**values)
    except FileNotFoundError:
        st.error(f"Error: `db/models.json` not found. Please create it.")
        return None
    except Exception as e:
        st.error(f"Error parsing `db/models.json`: {e}")
        return None
        
    return config

# --- Calculation Logic Functions ---

def calculate_memory_constraint(
    model: ModelConfig,
    gpu: GpuConfig,
    precision: str,
    max_batch_size: int,
    tokens_per_request: int,
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

    # Memory for the KV Cache (Llama-style rule of thumb)
    # Formula: 320 * seq_len * batch_size -> in MB, so / 1024 to convert to GB
    kv_cache_mem_gb = (320 * tokens_per_request * max_batch_size) / (2**20) / 1024

    total_mem_gb = model_mem_gb + activations_mem_gb + kv_cache_mem_gb
    
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

    # GPU throughput: how many "batch steps" (one token per sequence) can it perform per second?
    # This is equivalent to the tokens/second/sequence the GPU can generate.
    gpu_tps_per_sequence = gpu.flops_tflops[precision] / flops_per_batch_step

    logging.info(f"Throughput per GPU: {gpu_tps_per_sequence:.2f} tokens/sec/sequence")
    
    # If the GPU's throughput is less than the required per-user throughput, it's impossible.
    if tps_per_user > gpu_tps_per_sequence:
        return float("inf")

    # How many "user groups" (of size tps_per_user) can be served on one GPU?
    # This represents how many batching rounds can be time-multiplexed.
    rounds_per_gpu = floor(gpu_tps_per_sequence / tps_per_user)
    if not rounds_per_gpu:
        return float("inf")

    # Total number of batches to process to serve all users
    total_rounds_needed = ceil(users / max_batch_size)

    # The number of GPUs is the total rounds needed divided by the rounds one GPU can handle.
    return total_rounds_needed / rounds_per_gpu


# --- Streamlit User Interface ---

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(page_title="GpuCalculator", page_icon="💻")

    # Database of available configurations
    APP_CONFIG = load_app_config()
    if not APP_CONFIG:
        st.error("Error: Failed to load configuration. Exiting.")
        st.stop()
    
    st.title("LLM Inference GPU Calculator")
    st.markdown(
        "Estimate the number of GPUs required to serve a Large Language Model (LLM) "
        "using a *batching* strategy."
    )
    st.markdown("---")

    # --- Section 1: Define Your Workload ---
    st.subheader("1. Define Your Workload")
    users = st.number_input(
        label="Number of Concurrent Users",
        min_value=1,
        step=1,
        value=100,
        help="The total number of users the system must serve simultaneously.",
    )
    tps_per_user = st.number_input(
        label="Throughput per User (tokens/s)",
        min_value=1,
        step=1,
        value=10,
        help="The required token generation speed for each user.",
    )

    # --- Section 2: Select Your Model ---
    st.subheader("2. Select Your Model")
    model_key = st.selectbox(
        label="Choose a Model",
        options=list(APP_CONFIG["models"].keys()),
        format_func=lambda key: APP_CONFIG["models"][key].display_name,
        help="Choosing the model automatically determines its size."
    )
    selected_model = APP_CONFIG["models"][model_key]
        
    precision = st.selectbox(
        label="Quantization Precision",
        options=list(BYTES_PER_PARAM.keys()),
        help="The numerical precision of the model's weights. Lower precision reduces memory usage but may affect quality.",
    )
    tokens_per_req = st.number_input(
        label="Context Size (tokens/request)",
        min_value=1,
        step=1,
        value=2048,
        help="The maximum sequence length (context + generation) to be processed per request.",
    )

    # --- Section 3: Choose Hardware and Batching Strategy ---
    st.subheader("3. Choose Hardware and Batching Strategy")
    gpu_key = st.selectbox(
        label="Choose a GPU Type",
        options=list(APP_CONFIG["gpus"].keys()),
        format_func=lambda key: APP_CONFIG["gpus"][key].display_name,
        help="The hardware on which the model will run."
    )
    selected_gpu = APP_CONFIG["gpus"][gpu_key]
    max_batch = st.number_input(
        label="Maximum Batch Size",
        min_value=1,
        step=1,
        value=8,
        help="The number of user requests processed simultaneously in a single batch.",
    )

    st.markdown("---")

    # --- Calculation and Results Display ---
    if st.button("🚀 Calculate Required GPUs", type="primary", use_container_width=True):
        
        mem_constraint = calculate_memory_constraint(
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
            max_batch_size=max_batch,
            tokens_per_request=tokens_per_req,
        )

        speed_constraint = calculate_speed_constraint(
            users=users,
            tps_per_user=tps_per_user,
            max_batch_size=max_batch,
            model=selected_model,
            gpu=selected_gpu,
            precision=precision,
        )
        
        if speed_constraint == float("inf"):
            st.error(
                "**Impossible Configuration.** The requested throughput per user is too high for a single GPU "
                "with the current setup. Try reducing the batch size, choosing a more powerful GPU, "
                "or lowering the required throughput."
            )
        else:
            required_gpus = max(mem_constraint, speed_constraint)
            
            st.success(f"### Estimated Number of GPUs Required: **{ceil(required_gpus)}**")
            
            st.write("The calculation is based on the maximum of the following two constraints:")
            
            st.info(
                f"**Memory Constraint:** `{mem_constraint:.2f}` GPUs\n\n"
                "This is the minimum number of GPUs required to store the model and its caches (activations, KV cache)."
            )
            st.info(
                f"**Speed Constraint:** `{speed_constraint:.2f}` GPUs\n\n"
                "This is the number of GPUs required to meet the token throughput demand for all users."
            )
            
            if mem_constraint > speed_constraint:
                st.warning("Note: The bottleneck is **VRAM Memory**. The GPUs will be underutilized in terms of computation (FLOPS).")
            else:
                 st.warning("Note: The bottleneck is the **computation speed (FLOPS)**. The available VRAM on the GPUs will be more than sufficient.")


if __name__ == "__main__":
    main()
