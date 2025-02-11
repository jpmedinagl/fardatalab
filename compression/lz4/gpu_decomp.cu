#include <iostream>
#include <fstream>
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
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    data = new char[size];
    file.read(data, size);
    file.close();
}

void decompress_example(char* compressed_data, size_t compressed_size)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    char* device_compressed_data;
    CUDA_CHECK(cudaMalloc(&device_compressed_data, compressed_size));
    CUDA_CHECK(cudaMemcpyAsync(device_compressed_data, compressed_data, compressed_size, cudaMemcpyHostToDevice, stream));

    const size_t chunk_size = 65536;
    size_t num_chunks = (compressed_size + chunk_size - 1) / chunk_size;

    void** host_compressed_ptrs;
    CUDA_CHECK(cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * num_chunks));
    for (size_t i = 0; i < num_chunks; ++i) {
        host_compressed_ptrs[i] = device_compressed_data + chunk_size * i;
    }
    
    void** device_compressed_ptrs;
    CUDA_CHECK(cudaMalloc(&device_compressed_ptrs, sizeof(void*) * num_chunks));
    CUDA_CHECK(cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));

    size_t* device_compressed_bytes;
    CUDA_CHECK(cudaMalloc(&device_compressed_bytes, sizeof(size_t) * num_chunks));
    CUDA_CHECK(cudaMemcpyAsync(device_compressed_bytes, &compressed_size, sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));

    size_t* device_uncompressed_bytes;
    CUDA_CHECK(cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * num_chunks));

    size_t temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(num_chunks, chunk_size, &temp_bytes);
    void* device_temp_ptr;
    CUDA_CHECK(cudaMalloc(&device_temp_ptr, temp_bytes));

    size_t uncompressed_size = num_chunks * chunk_size;
    char* device_uncompressed_data;
    CUDA_CHECK(cudaMalloc(&device_uncompressed_data, uncompressed_size));

    void** host_uncompressed_ptrs;
    CUDA_CHECK(cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * num_chunks));
    for (size_t i = 0; i < num_chunks; ++i) {
        host_uncompressed_ptrs[i] = device_uncompressed_data + chunk_size * i;
    }
    void** device_uncompressed_ptrs;
    CUDA_CHECK(cudaMalloc(&device_uncompressed_ptrs, sizeof(void*) * num_chunks));
    CUDA_CHECK(cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));

    nvcompStatus_t* device_statuses;
    CUDA_CHECK(cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * num_chunks));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
        (const void *const *)device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        device_uncompressed_bytes,
        num_chunks,
        device_temp_ptr,
        temp_bytes,
        (void *const *)device_uncompressed_ptrs,
        device_statuses,
        stream
    );
    
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    if (decomp_res != nvcompSuccess) {
        std::cerr << "GPU decompression failed!" << std::endl;
        return;
    }

    std::cout << "GPU decompression time: " << gpuTime << " ms\n";
    float compressionRatio = (float)compressed_size / uncompressed_size;
    std::cout << "Compression Ratio: " << compressionRatio << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cout << "usage: ./gpu_decomp <compressed_file>\n";
        return 1;
    }

    const char* filename = argv[1];
    char* compressed_data = nullptr;
    size_t compressed_size = 0;

    read_file_data(filename, compressed_data, compressed_size);

    std::cout << "Starting GPU decompression..." << std::endl;
    decompress_example(compressed_data, compressed_size);

    delete[] compressed_data;
    return 0;
}
