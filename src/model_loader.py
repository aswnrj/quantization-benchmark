import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_id, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device, attn_implementation="flex_attention")
    model.eval()
    return (model, tokenizer)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model, tokenizer = load_model_and_tokenizer('meta-llama/Llama-3.2-1B', torch.float16, device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} vocab: {len(tokenizer):,}")
