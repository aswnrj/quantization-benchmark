import torch
import time
import statistics
from src import model_loader

def time_callable(fn, n_runs=5, n_warmup=2):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for time measurement")
    for _ in range(n_warmup):
        fn()
    timings = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)
    return timings

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available. Run on Google Colab")
        raise SystemExit(0)
    device = 'cuda'
    model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', torch.float16, device)
    
    prompt = "The Llama 3 model architecture differs from earlier transformers in several ways:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    timings = time_callable(lambda: model(input_ids))
    print(f"Timings for the runs: {timings}")
    print(f"Median: {statistics.median(timings)}, Percentiles: {statistics.quantiles(timings, n=4)}")