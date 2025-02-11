#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <nvcomp/lz4.h>

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void read_file_data(const char* filename, char*& data, size_t& size)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the file size and allocate buffer for data
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    data = new char[size];

    // Read the file content into the buffer
    file.read(data, size);
    file.close();
}

void write_file_data(const char* filename, char* data, size_t size)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    file.write(data, size);
    file.close();
}

void compression(char* input_data, const size_t in_bytes)
{
    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Define chunk size and batch size
    const size_t chunk_size = 65536;
    const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

    // Allocate device memory
    char* device_input_data;
    CUDA_CHECK(cudaMalloc(&device_input_data, in_bytes));
    CUDA_CHECK(cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream));

    // Set up uncompressed data pointers and sizes
    size_t* host_uncompressed_bytes;
    CUDA_CHECK(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size));
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_bytes[i] = (i + 1 < batch_size) ? chunk_size : in_bytes - (chunk_size * i);
    }

    void** host_uncompressed_ptrs;
    CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * batch_size));
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = device_input_data + chunk_size * i;
    }

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;
    CUDA_CHECK(cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size));
    CUDA_CHECK(cudaMalloc(&device_uncompressed_ptrs, sizeof(void*) * batch_size));
    CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream));

    // Allocate space for temporary memory and compressed data
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    void** host_compressed_ptrs;
    CUDA_CHECK(cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * batch_size));
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMalloc(&host_compressed_ptrs[i], max_out_bytes));
    }

    void** device_compressed_ptrs;
    CUDA_CHECK(cudaMalloc(&device_compressed_ptrs, sizeof(void*) * batch_size));
    CUDA_CHECK(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream));

    size_t* device_compressed_bytes;
    CUDA_CHECK(cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size));

    // Start the timer for GPU compression
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform GPU compression using nvcomp
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_compressed_ptrs,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream
    );

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    if (comp_res != nvcompSuccess) {
        std::cerr << "GPU compression failed!" << std::endl;
        return;
    }

    std::cout << "GPU compression time: " << gpuTime << " ms\n";
    float gpuThroughput = (in_bytes / (1024.0f * 1024.0f)) / (gpuTime / 1000.0f);
    std::cout << "GPU Throughput: " << gpuThroughput << " MB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Retrieve the compressed data from device to host
    char* compressed_data = new char[batch_size * max_out_bytes];
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMemcpy(compressed_data + i * max_out_bytes, host_compressed_ptrs[i], max_out_bytes, cudaMemcpyDeviceToHost));
    }

    // Write compressed data to a file
    write_file_data("temp", compressed_data, batch_size * max_out_bytes);

    // Clean up
    delete[] compressed_data;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char ** argv)
{
    if (argc != 2) {
        std::cout << "usage: ./gpu_comp <file>\n";
        return 1;
    }

    const char* filename = argv[1];  // Change this to your data file
    char* uncompressed_data = nullptr;
    size_t data_size = 0;

    // Read the data from the file
    read_file_data(filename, uncompressed_data, data_size);

    // Run GPU compression
    std::cout << "Starting GPU compression..." << std::endl;
    compression(uncompressed_data, data_size);

    // Clean up
    delete[] uncompressed_data;

    return 0;
}
