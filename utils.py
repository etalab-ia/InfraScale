# utils.py

BYTES_PER_PARAM = {
    "8 bits (int8/fp8)": 1,
    "16 bits (fp16)": 2,
    "32 bits (fp32)": 4
}

GPU_EFFICIENCY_FACTORS = {
    "8 bits (int8/fp8)": 0.45,
    "16 bits (fp16)": 0.55,
    "32 bits (fp32)": 0.35
}

def get_model_layers(model):
    """Estimate number of layers if not explicitly defined in the model config."""
    return model.n_layers if model.n_layers else int(model.size_b_params**0.5 * 8)
