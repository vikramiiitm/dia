import torch

def load_model_optimized():
    from transformers import BitsAndBytesConfig
    from dia.model import DiaModel

    print("Loading Dia in 8-bit mode...")

    quant = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,  # CRITICAL FOR 4GB VRAM
    )

    model = DiaModel.from_pretrained(
        "nari-labs/dia",
        device_map={"": "cuda"},            # main layers on GPU
        quantization_config=quant,          # 8-bit
        torch_dtype=torch.float16,          # remaining layers
        low_cpu_mem_usage=True,
    )

    torch.cuda.empty_cache()

    print("Dia loaded in 8-bit (VRAM-safe for GTX 1650)")
    return model
