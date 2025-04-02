import os
import re
import csv

# Define the base directory for your benchmarks
col = "6"
base_dir = "/mnt/dataset/benchmark/col" + col
starts = "bin"
output_csv = "benchmark_" + starts + ".csv"

# Initialize a list to store the results
results = []

# Regular expressions to extract the needed data
regex = {
    "uncompressed": r"uncompressed \(B\): (\d+)",
    "comp_size": r"comp_size: (\d+)",
    "comp_ratio": r"compressed ratio: ([\d.]+)",
    "compression_throughput": r"compression throughput \(GB/s\): ([\d.]+)",
    "decompression_throughput": r"decompression throughput \(GB/s\): ([\d.]+)"
}

# Loop through each experiment folder (e.g., 'deflate_chunked', 'bitcomp_chunked')
for experiment in os.listdir(base_dir):
    experiment_dir = os.path.join(base_dir, experiment)
    if os.path.isdir(experiment_dir):
        # Loop through each file in the experiment folder
        for file in os.listdir(experiment_dir):
            if file.startswith(starts) and file.endswith(".txt"):
                # Extract the scale (filename without .txt)
                scale = file.replace(".txt", "")
                scale = scale.replace(starts, "")

                # Open the file and read its contents
                with open(os.path.join(experiment_dir, file), 'r') as f:
                    data = f.read()

                    # Extract values using the regular expressions
                    extracted_data = {
                        "benchmark": experiment,
                        "scale": scale
                    }
                    for key, pattern in regex.items():
                        match = re.search(pattern, data)
                        if match:
                            extracted_data[key] = match.group(1)
                        else:
                            extracted_data[key] = ""  # If data is missing

                    # Append the extracted data to the results list
                    results.append(extracted_data)

# Write the results to a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["benchmark", "scale", "uncompressed", "comp_size", "comp_ratio", "compression_throughput", "decompression_throughput"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Results have been written to {output_csv}")
