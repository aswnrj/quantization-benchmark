import json
import matplotlib.pyplot as plt

CACHE_TYPES = ['fp16', 'int8', 'int4']
SEQ_LEN_INTS = [512, 1024, 2048, 4096, 8192, 16384]
SEQ_LEN_LABELS = ['512', '1K', '2K', '4K', '8K', '16K']
COLORS = {'fp16': 'C0', 'int8': 'C1', 'int4': 'C2'}

def plot_metric(metric_dict, ylabel, title, output_path, transform=lambda v:v, yscale='linear', start_idx=0):
    seq_lens = SEQ_LEN_INTS[start_idx:]
    seq_len_labels = SEQ_LEN_LABELS[start_idx:]
    fig, ax = plt.subplots(figsize=(8, 5))
    for ct, values in metric_dict.items():
        ys = [transform(v) for v in values[start_idx:]]
        ax.plot(seq_lens, ys, marker='o', label=ct.upper(), color=COLORS[ct])
    ax.set_xscale('log', base=2)
    ax.set_xticks(seq_lens)
    ax.set_xticklabels(seq_len_labels)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_path}')

def get_fields_from_data(data, field):
    return {
        ct: [cell[field] for cell in data[ct]['results'].values()]
        for ct in CACHE_TYPES
    }

if __name__ == "__main__":
    data = {}
    for ct in CACHE_TYPES:
        with open(f'results/benchmark_{ct}.json', 'r') as file:
            data[ct] = json.load(file)
            
    plot_metric(
        get_fields_from_data(data, 'peak_memory_bytes'),
        ylabel='Peak GPU memory (GiB)',
        title='Peak memory vs sequence length',
        output_path='results/plots/memory_vs_seqlen.png',
        transform=lambda v: v / (1024**3),
    )
    
    plot_metric(
        get_fields_from_data(data, 'decode_tok_per_sec'),
        ylabel='Decode throughput (tokens/s)',
        title='Decode throughput vs sequence length (4K+; smaller contexts noise-dominated)',
        output_path='results/plots/decode_vs_seqlen.png',
        start_idx=3,
    )

    plot_metric(
        get_fields_from_data(data, 'prefill_tok_per_sec'),
        ylabel='Prefill throughput (tokens/s)',
        title='Prefill throughput vs sequence length (log y)',
        output_path='results/plots/prefill_vs_seqlen.png',
        yscale='log',
    )
