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
        window = input_ids[:, begin:end]
        labels = window.clone()
        labels[:, :-score_len] = -100
        prev_end = end
        if score_len < 0:
            break

        cache = cache_factory() if cache_factory else None
        with torch.inference_mode():
            output = model(input_ids=window, past_key_values=cache, use_cache=True)

        shift_logits = output.logits[:, :-1, :]             # (1, T-1, C)
        shift_labels = labels[:, 1:]                        # (1, T-1)
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[2]), shift_labels.reshape(-1), ignore_index=-100, reduction='sum')
        n_scored = (shift_labels != -100).sum().item()

        total_nll += loss.item()
        total_tokens += n_scored
        n_windows += 1
        
        del cache, output, shift_logits, shift_labels
        if end == T:
            break
    
    ppl = math.exp(total_nll / total_tokens)
    return {'perplexity': ppl, 'n_tokens_scored': total_tokens, 'n_windows': n_windows}