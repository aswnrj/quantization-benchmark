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

## KV Cache Memory Formula 

$$ 2 * n\\_layers * n\\_kv\\_heads * head\\_dim * seq\\_len * batch\\_size * dtype\\_bytes  $$

- 2 signifies key and value tensors. Separate memory for both
- `n_layers`: Each transformer layer has its own set of K and V tensors
- `n_kv_heads`: This is 8 for Llama3.2-1B due to GQA, even though there are 32 Q heads per layer
- `head_dim`: The number of dimensions in each head
- `seq_len`: This grows as the context grows
- `batch_size`: We will assume it to be 1 here
- `dtype_bytes`: FP16 = 2, INT8 = 1, INT4 = 0.5

### Per-token cost for Llama3.2-1B (FP16)
$$  2 * 16 * 8 * 64 * seq\\_len * 1 * 2 = 32,768 * seq\\_len $$

Size of KV cache per token is $32,768$ bytes

### Cache size across context lengths

| Context length | Size |
| -------------- | ---- |
| 1K | 32MiB |
| 4K | 128MiB |
| 16K | 512MiB |
| 64K | 2GiB |
| 128K | 4GiB |

### When does cache dominate weights?
The total size of weights for this model is $2.47GB$. A context size of $75K$ is enough to occupy that much, post which the cache size dominates the overall size of the model.
 

## Quantization

### Why quantize the KV cache?
For models with long contexts, KV cache keeps growing and after a point (75k tokens in Llama3.2-1B), it grows larger than the model weights. This is why quantization is required. 

### Affine quantization
Float range: $[x_{min}, x_{max}]$
Int range: $[q_{min}, q_{max}]$
$$q = \text{round}(x / s + z), \quad x_{\text{deq}} = s \cdot (q - z)$$
$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}, \quad z = \text{round}\Big(q_{\min} - \frac{x_{\min}}{s}\Big)$$
$z$: the integer that maps to float 0

### Symmetric vs Asymmetric
- Symmetric fixes zero point `z=0`
- Good for data centered around origin (post-RMSNorm + linear, KV cache is roughly centered around origin)
- Can waste range if data is not centered (eg. ReLU)
- Asymmetric shifts the quantized range to match distribution
- Better accuracy for skewed data
- Slightly more complex, needs zero-point adjustments

### Bit width
| Scheme | levels |	step size (relative) |	error variance (relative) |
| -----| ----- | ----- | ----- |
| FP16 |	~65k|	—	|~0|
| INT8 |	256	|1×	|1×|
| INT4 | 16	|16×|	256×|

INT4 will have 256 times more error variance as INT8 as the error variance scales as $s^2$

### Granularity
- The granularity can be per-tensor, per-channel, per-token and per-group ranging from worst to best quality.
- Per-token is the natural fit for KV cache (each token's K/V is computed once when generated)
- The tradeoff: finer granularity = better quality, more metadata.
- The mechanism: finer granularity -> smaller per-group range -> smaller s-> smaller error
- HQQ uses per group for INT4 with group size of 64

### Effective bytes per element
- We usually ignore the metadata (s and z) when we talk about bytes per element. This makes it look like INT4 gives 4x improvement as FP16.
- But in reality, due to metadata, INT4 with group size of 64 is only 3.56x more efficient than FP16.
- `bytes/elem = raw + metadata_bytes / group_size`
- If we consider s and z are fp16 and occupy 2 bytes each and we need 1s and 1z per 64 values: INT4 will need $0.5 + 4/64 = 0.5625$ bytes per value, which is a 3.56x improvement from fp16 which needs 2 bytes.

### Quantization error
$$\text{Var}(\epsilon) = s^2/12$$
Rounding to nearest integer with step size s produces noise on `[-s/2, +s/2]`
Hence, finer granularity helps. 

### Why KV cache is harder than weights
- Weights are static, but KV cache keeps changing with each token
- K has outliers, but V is fine. Hence we have methods like KIVI which quantize K per channel and V per token
- Attention is sensitive to small changes. Near-tie scores can flip when there are small errors in K. 