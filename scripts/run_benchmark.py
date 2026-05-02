import torch
import json
from src import model_loader, runner

def run_benchmark(sweep_list):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for memory measurement")
    device = "cuda"
    model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', dtype=torch.float16, device=device)
    metrics = {}
    for name, seq_len in sweep_list:
        metrics[name] = runner.run_cell(model, tokenizer, seq_len)
        with open('results/benchmark.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"  saved through {name}")
    return metrics

if __name__ == "__main__":
    sweep_list = [('512', 512), ('1k', 1024), ('2k', 2048), ('4k', 4096), ('8k', 8192), ('16k', 16384)]
    metrics = run_benchmark(sweep_list)