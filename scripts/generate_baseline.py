import torch
from src import model_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
prompt = "The Llama 3 model architecture differs from earlier transformers in several ways:"

model, tokenizer = model_loader.load_model_and_tokenizer('meta-llama/Llama-3.2-1B', torch.float16, device)
inputs = tokenizer(prompt, return_tensors='pt').to(device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

text_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(text_output)

