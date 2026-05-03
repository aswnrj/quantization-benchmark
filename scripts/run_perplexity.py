import torch
import json
from src import model_loader
from datasets import load_dataset
from transformers import QuantizedCache
from src import perplexity

def run_perplexity():
    dataset = "wikitext"; subset = "wikitext-2-raw-v1"; split = "test" 
    max_length = 2048; stride = 512
    model_name = "meta-llama/Llama-3.2-1B"; model_dtype = torch.float16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer = model_loader.load_model_and_tokenizer(model_name, dtype=model_dtype, device=device)

    ds = load_dataset(dataset, subset, split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())          # concatenate all rows

    variants = [
        ("fp16", None, {}),
        ("int8", lambda: QuantizedCache(backend='hqq', config=model.config, nbits=8, axis_key=0, axis_value=0, q_group_size=64, residual_length=64), {'nbits':8, 'q_group_size':64, 'residual_length':64}),
        ("int4", lambda: QuantizedCache(backend='hqq', config=model.config, nbits=4, axis_key=0, axis_value=0, q_group_size=64, residual_length=64), {'nbits':4, 'q_group_size':64, 'residual_length':64})
    ]

    meta = { 
        'dataset':dataset, 'split': split, 
        'max_length': max_length, 'stride': stride, 
        'model_name': model_name, 'model_dtype': str(model_dtype) 
    }

    results = {}
    for name, factory, cache_config in variants:
        print(f"Running {name}...")
        out = perplexity.compute_perplexity(model, tokenizer, text, factory, max_length, stride)
        results[name] = {**out, "cache_config": cache_config}
        # Write JSON after each variant to save partial progress
        json.dump({"meta": meta, "results": results}, open("results/perplexity.json", "w"), indent=4)

if __name__ == "__main__":
    run_perplexity()