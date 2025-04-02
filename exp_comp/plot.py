import pandas as pd
import matplotlib.pyplot as plt

def plot_key(df, key):
    df = df.sort_values(by='scale')

    plt.figure(figsize=(12, 8))

    for benchmark in df['benchmark'].unique():
        subset = df[df['benchmark'] == benchmark]
        plt.plot(subset['scale'], subset[key], label=benchmark, marker='o')

    plt.title(key + " (GB/s) vs Size")
    plt.xlabel("Size (scale)")
    plt.ylabel(key + " (GB/s)")
    # plt.yscale('log')
    plt.legend(title="Benchmark", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(key)


if __name__ == '__main__':
    keys = ["compression_throughput", "decompression_throughput"] #, "comp_ratio", ]

    df = pd.read_csv("benchmark_results.csv")

    for key in keys:
        plot_key(df, key)