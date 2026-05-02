import torch
from src import model_loader

def reset_peak_memory(device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for memory measurement")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

def peak_memory_bytes(device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for memory measurement")
    return torch.cuda.max_memory_allocated(device)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found. Run on Colab")
        raise SystemExit(0)
    device = 'cuda'
    model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', torch.float16, device)
    
    prompt = "The Llama 3 model architecture differs from earlier transformers in several ways:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    reset_peak_memory()
    with torch.inference_mode():
        output = model(input_ids)
    print(f"Peak memory bytes: {peak_memory_bytes()/(1024**3):.2f}GiB")
    weight_memory = sum(p.numel()*p.element_size() for p in model.parameters())
    print(f"Weight memory: {weight_memory/(1024**3):.2f}GiB")
