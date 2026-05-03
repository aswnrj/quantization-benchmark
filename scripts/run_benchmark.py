import torch
import json
from src import model_loader, runner
from transformers import QuantizedCache

def run_benchmark(sweep_list, quantized=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for memory measurement")
    device = "cuda"
    model_name = 'meta-llama/Llama-3.2-1B'
    model, tokenizer = model_loader.load_model_and_tokenizer(model_name, dtype=torch.float16, device=device)
    meta = {
        'model_name': model_name,
        'model_dtype': 'float16',
        'cache_type': 'fp16',
    }
    cache_factory = None
    output_path = 'results/benchmark_fp16.json'

    if quantized == 'int8':
        nbits, q_group_size, residual_length = 8, 64, 64
        cache_factory = lambda: QuantizedCache(
                            backend='hqq', config=model.config,
                            nbits=nbits, axis_key=0, axis_value=0,
                            q_group_size=q_group_size, residual_length=residual_length
                        )
        meta.update({
            'cache_type': 'int8',
            'backend': 'hqq',
            'nbits': nbits,
            'q_group_size': q_group_size,
            'residual_length': residual_length,
            'axis_key': 0,
            'axis_value': 0,
        })
        output_path = 'results/benchmark_int8.json'
    elif quantized == 'int4':
        nbits, q_group_size, residual_length = 4, 64, 64
        cache_factory = lambda: QuantizedCache(
                            backend='hqq', config=model.config,
                            nbits=nbits, axis_key=0, axis_value=0,
                            q_group_size=q_group_size, residual_length=residual_length
                        )
        meta.update({
            'cache_type': 'int4',
            'backend': 'hqq',
            'nbits': nbits,
            'q_group_size': q_group_size,
            'residual_length': residual_length,
            'axis_key': 0,
            'axis_value': 0,
        })
        output_path = 'results/benchmark_int4.json'

    results = {}
    for name, seq_len in sweep_list:
        results[name] = runner.run_cell(model, tokenizer, seq_len, cache_factory)
        with open(output_path, 'w') as f:
            json.dump({'meta': meta, 'results': results}, f, indent=4)
        print(f"  saved through {name}")
    return results

if __name__ == "__main__":
    sweep_list = [('512', 512), ('1k', 1024), ('2k', 2048), ('4k', 4096), ('8k', 8192), ('16k', 16384)]
    results = run_benchmark(sweep_list, 'int4')