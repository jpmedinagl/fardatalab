#!/bin/bash

# Define the experiments to run
experiments=("ans" "bitcomp" "cascaded" "deflate" "gdeflate" "lz4" "snappy" "zstd")
args=("" "" "" "" "" "" "" "" "")

# Define the input sizes to loop through
sizes=(1 2 5 10 20 50)

# Base output directory
base_output_dir="/mnt/dataset/benchmark"

# Ensure base directory exists
mkdir -p "$base_output_dir"

# Loop through each experiment
for i in "${!experiments[@]}"; do
    exp="${experiments[i]}"
    exp_args="${args[i]}"
    
    # Loop through each size
    for size in "${sizes[@]}"; do
        # Define output file path
        output_file="${base_output_dir}/${exp}/${size}.txt"

        # Ensure experiment-specific directory exists
        mkdir -p "$(dirname "$output_file")"

        # Print status message
        echo "${exp} ${size} ${output_file} ${exp_args}"

        # Check if the input file exists before running
        input_file="/mnt/dataset/data/li${size}.bin"
        if [[ -f "$input_file" ]]; then
            ./benchmark_"${exp}"_chunked -f "$input_file" ${exp_args} > "$output_file" 2>&1
        else
            echo "File $input_file not found!" | tee -a "$output_file"
        fi
    done
done
