import torch
import time
import statistics
from src import model_loader

def time_callable(fn, n_runs=5, n_warmup=2):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for time measurement")
    for _ in range(n_warmup):
        with torch.inference_mode():
            fn()
    timings = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append(end - start)
    return timings, statistics.median(timings)

def time_prefill(model, input_ids, n_runs=5, n_warmup=2):
    return time_callable(lambda: model(input_ids, use_cache=True), n_runs, n_warmup)

def time_decode_step(model, cache, last_token, n_runs=5, n_warmup=2):
    # As we call this multiple times (n_runs + n_warmup), the cache side keeps increasing and timings might be slightly inaccurate due to this
    # But if seq_len is 100s to 1000s, the diff should be negligible
    return time_callable(lambda: model(last_token, past_key_values=cache, use_cache=True), n_runs, n_warmup)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU not available. Run on Google Colab")
        raise SystemExit(0)
    device = 'cuda'
    model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', torch.float16, device)
    
    prompt = "The Llama 3 model architecture differs from earlier transformers in several ways:"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.inference_mode():
        output = model(input_ids, use_cache=True)

    cache = output.past_key_values
    last_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    prefill_time, prefill_median = time_prefill(model, input_ids)
    decode_time, decode_median = time_decode_step(model, cache, last_token)

    print(f"Prefill step timings: {prefill_time}")
    print(f"Decode step timings: {decode_time}")
    
    prefill_toks = input_ids.shape[1]/prefill_median
    decode_toks = 1/decode_median

    print(f"Prefill tok/s: {prefill_toks:.4f}")
    print(f"Decode tok/s: {decode_toks:.4f}")

    print(f"Ratio of prefill tok/s to decode tok/s: {prefill_toks/decode_toks:.2f}")