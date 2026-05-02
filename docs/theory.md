# Theory Notes

## Llama 3.2-1B Architecture

### From config.json
| Field | Value | Why it matters |
| ----- | ----- | -------------- |
| num_hidden_layers | 16 | the L in our cache formula |
| num_attention_heads | 32 | Q heads, affects compute, not cache size |
| num_key_value_heads | 8 | the n_kv_heads in cache size |
| hidden_size | 2048 | total embedding dimension |
| head_dim | 64 | per head dimension |
| intermediate_size | 8192 | FFN width |
| vocab_size | 128256 | tokenizer vocab |
| max_position_embeddings | 131072 | max context (128K); bound for seq_len sweep |
| tie_word_embeddings | true | input/output embeddings share weights |
| torch_dtype | bfloat16 | this is why we explicitly override to float16 on T4 GPU |

### Grouped-Query Attention (GQA)
1. In Multi-Head Attention (MHA), each of the Q heads has their own K and V heads. 
1. In Multi-Query Attention (MQA), all the Q heads share one K head and one V head and this is broadcast to all `n_heads` Q heads. This is efficient, but comes at a cost of expressiveness.
1. Grouped Query Attention (GQA) is a middle ground. `n_kv_heads` number of K heads and V heads are shared among `n_heads` query heads and each K and V head is broadcasted to all the Q heads of that group. 
1. This was created because the long-context inference in MHA made KV cache memory dominant. KV cache size was earlier `2 * batch_size * n_layers * n_heads * head_dim * seq_len * dtype_bytes`. By using GQA, we can reduce it by a factor of `n_heads / n_kv_heads`.
1. For Llama3.2-1B model, the `n_heads` is 32. The `n_kv_heads` is 8 which makes it 4x more efficient than using MHA and the quality of the model is comparable. We use `n_kv_heads` in the cache size calculation formula now, instead of `n_heads` earlier. 