import pandas as pd
import matplotlib.pyplot as plt

def plot_key(df, key):
    df = df.sort_values(by='scale')

    plt.figure(figsize=(10, 6))

    for benchmark in df['benchmark'].unique():
        subset = df[df['benchmark'] == benchmark]
        plt.plot(subset['scale'], subset[key], label=benchmark, marker='o')

    plt.title(key + " vs Size")
    plt.xlabel("Size (scale)")
    plt.ylabel(key)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(title="Benchmark", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(key)


if __name__ == '__main__':
    keys = ["comp_ratio"] #, "compression_throughput", "decompression_throughput"] #, "comp_ratio", ]

    df = pd.read_csv("benchmark_results.csv")

    for key in keys:
        plot_key(df, key)
