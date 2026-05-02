import torch
from src import memory, timing

def run_cell(model, tokenizer, seq_len, n_runs=10, n_warmup=3):
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=model.device)
    
    memory.reset_peak_memory()
    with torch.inference_mode():
        output = model(input_ids, use_cache=True, logits_to_keep=1)
    
    cache = output.past_key_values
    last_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    print(f"Running seq_len: {seq_len}")
    peak_memory_bytes = memory.peak_memory_bytes()
    print(f"Peak memory bytes: {peak_memory_bytes/(1024**3):.2f}GiB")
    kv_cache_size_observed = sum(layer.keys.numel()*layer.keys.element_size() + layer.values.numel()*layer.values.element_size() for layer in cache.layers)
    print(f"Observed KV cache size: {kv_cache_size_observed/(1024**2):.2f}MiB")
    kv_cache_size_theoretical = 2*model.config.num_hidden_layers*model.config.num_key_value_heads*model.config.head_dim*seq_len*cache.layers[0].keys.element_size()
    print(f"Calculated KV cache size: {kv_cache_size_theoretical/(1024**2):.2f}MiB")

    assert kv_cache_size_observed == kv_cache_size_theoretical

    prefill_timings, prefill_median = timing.time_prefill(model, input_ids, n_runs, n_warmup)
    decode_timings, decode_median = timing.time_decode_step(model, cache, last_token, n_runs, n_warmup)

    prefill_toks = input_ids.shape[1]/prefill_median
    decode_toks = 1/decode_median

    del cache, output, last_token, input_ids
    torch.cuda.empty_cache()
    return {
        'prefill_timings_ms': [t*1000 for t in prefill_timings],
        'decode_timings_ms': [t*1000 for t in decode_timings],
        'prefill_median_timing_ms': prefill_median*1000,
        'decode_median_timing_ms': decode_median*1000,
        'prefill_tok_per_sec': prefill_toks,
        'decode_tok_per_sec': decode_toks,
        'peak_memory_bytes': peak_memory_bytes,
        'kv_cache_observed_bytes': kv_cache_size_observed,
        'kv_cache_theoretical_bytes': kv_cache_size_theoretical
    }