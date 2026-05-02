import torch
from src import model_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', dtype=torch.float16, device=device)
prompt = "The Llama 3 model architecture differs from earlier transformers in several ways:"

inputs = tokenizer(prompt, return_tensors='pt').to(device)

with torch.inference_mode():
    output = model(**inputs, use_cache=True)

cache = output.past_key_values
print(f"Cache type: {type(cache)}, Num layers: {len(cache)}, Seq len: {cache.get_seq_length()}")
print(f"Keys shape: {cache.layers[0].keys.shape}, Values shape: {cache.layers[0].values.shape}")

total_bytes = sum(layer.keys.numel()*layer.keys.element_size() + layer.values.numel()*layer.values.element_size() for layer in cache.layers)
n_layers = model.config.num_hidden_layers
n_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim
seq_len = cache.get_seq_length()
dtype_bytes = cache.layers[0].keys.element_size()
total_bytes_from_eq = 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes

assert total_bytes == total_bytes_from_eq       # 524,288
print(f"From tensors: {total_bytes:,}bytes, From equation: {total_bytes_from_eq:,}bytes")

next_id = output.logits[:, -1, :].argmax(-1, keepdim=True)

with torch.inference_mode():
    next_output = model(next_id, past_key_values=cache, use_cache=True)

next_cache = next_output.past_key_values
print(f"New shape of keys and values: {next_cache.layers[0].keys.shape}")

print(f"Decoded word: {tokenizer.decode([next_id.item()], clean_up_tokenization_spaces=False)}")