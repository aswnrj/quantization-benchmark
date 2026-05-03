import torch
import torch.nn.functional as F
import math

def compute_perplexity(model, tokenizer, prompt, cache_factory=None, max_length=2048, stride=512):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    T = input_ids.shape[1]
    total_nll, total_tokens, n_windows = 0.0, 0, 0
    prev_end = 0
    for begin in range(0, T, stride):
        end = min(begin + max_length, T)
        score_len = end - prev_end
        if score_len <= 0:
            break
        window = input_ids[:, begin:end]
        L = window.shape[1]
        prefix_len = L - score_len
        prefix = window[:, :prefix_len]
        suffix = window[:, prefix_len:]
        prev_end = end

        # Two-pass eval so the quantized cache is actually read during attention.
        # Pass 1 fills the cache with the prefix (no logits needed).
        # Pass 2 evaluates the suffix against attention that reads the (quantized) prefix from cache.
        cache = cache_factory() if cache_factory else None
        with torch.inference_mode():
            if prefix.shape[1] > 0:
                out1 = model(input_ids=prefix, past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = out1.past_key_values
                del out1
            output = model(input_ids=suffix, past_key_values=cache, use_cache=True)

        shift_logits = output.logits[:, :-1, :]
        shift_labels = suffix[:, 1:]
        V = shift_logits.shape[-1]
        loss = F.cross_entropy(
            shift_logits.float().reshape(-1, V),
            shift_labels.reshape(-1),
            reduction='sum',
        )
        n_scored = shift_labels.numel()

        total_nll += loss.item()
        total_tokens += n_scored
        n_windows += 1

        del cache, output, shift_logits, shift_labels
        if end == T:
            break

    ppl = math.exp(total_nll / total_tokens)
    return {'perplexity': ppl, 'n_tokens_scored': total_tokens, 'n_windows': n_windows}
