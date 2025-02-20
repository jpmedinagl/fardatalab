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

int compress_chunk(char* input_data_chunk, const size_t chunk_size, cudaStream_t stream) 
{
    char* device_input_data;
    CUDA_CHECK(cudaMalloc(&device_input_data, chunk_size));
    CUDA_CHECK(cudaMemcpy(device_input_data, input_data_chunk, chunk_size, cudaMemcpyHostToDevice));

    size_t* host_uncompressed_bytes;
    CUDA_CHECK(cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t)));
    host_uncompressed_bytes[0] = chunk_size;

    void** host_uncompressed_ptrs;
    CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*)));
    host_uncompressed_ptrs[0] = device_input_data;

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;
    CUDA_CHECK(cudaMalloc(&device_uncompressed_bytes, sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&device_uncompressed_ptrs, sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*), cudaMemcpyHostToDevice));

    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(1, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    void* device_compressed_ptr;
    CUDA_CHECK(cudaMalloc(&device_compressed_ptr, max_out_bytes));

    size_t* device_compressed_bytes;
    CUDA_CHECK(cudaMalloc(&device_compressed_bytes, sizeof(size_t)));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        1,
        device_temp_ptr,
        temp_bytes,
        &device_compressed_ptr,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream
    );

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    if (comp_res != nvcompSuccess) {
        std::cerr << "GPU compression failed!" << std::endl;
        return -1;
    }

    std::cout << "GPU compression time for chunk: " << gpuTime << " ms\n";
    float gpuThroughput = (chunk_size / (1024.0f * 1024.0f)) / (gpuTime / 1000.0f);
    std::cout << "GPU Throughput: " << gpuThroughput << " MB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_input_data));
    CUDA_CHECK(cudaFree(device_temp_ptr));
    CUDA_CHECK(cudaFree(device_compressed_ptr));
    CUDA_CHECK(cudaFree(device_compressed_bytes));
    CUDA_CHECK(cudaFree(device_uncompressed_bytes));
    CUDA_CHECK(cudaFree(device_uncompressed_ptrs));

    return 0;
}

int compression(char* input_data, const size_t in_bytes) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t chunk_size = 65536;
    const size_t num_chunks = (in_bytes + chunk_size - 1) / chunk_size;

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        size_t chunk_offset = chunk_idx * chunk_size;
        size_t chunk_data_size = std::min(chunk_size, in_bytes - chunk_offset);

        if (compress_chunk(input_data + chunk_offset, chunk_data_size, stream) == -1) {
            return -1;
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}

int main(int argc, char ** argv)
{
    if (argc != 2) {
        std::cout << "usage: ./gpu_comp <file>\n";
        return 1;
    }

    const char* filename = argv[1];
    char* uncompressed_data = nullptr;
    size_t data_size = 0;

    read_file_data(filename, uncompressed_data, data_size);

    std::cout << "Starting GPU compression..." << std::endl;
    compression(uncompressed_data, data_size);

    delete[] uncompressed_data;

    return 0;
}
