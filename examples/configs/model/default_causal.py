from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map: str, max_memory: dict = None, torch_dtype: str = "auto"):
    import torch
    if max_memory is not None:
        max_memory = {int(k) if str(k).isdigit() else k: v 
                      for k, v in max_memory.items()}
    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()
    return model

def load_tokenizer(model_path: str, add_bos_token: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_bos_token=add_bos_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
