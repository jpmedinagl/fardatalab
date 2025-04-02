#include <cuda_runtime.h>
#include <iostream>

#define SIZE 1024
#define BATCH 256

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

int main() 
{
    int *d_src, *d_dst;  // Device pointers
    int *h_src, *h_dst;  // Host buffers

    h_src = new int[SIZE];
    h_dst = new int[SIZE];

    // Random data
    for (int i = 0; i < SIZE; i++) {
        h_src[i] = i;
    }

    // Set up GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&d_src, SIZE * sizeof(int)));

    // Copy data to GPU 0
    CUDA_CHECK(cudaMemcpy(d_src, h_src, SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Set up GPU 1
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, SIZE * sizeof(int)));

    
    // Transfer GPU 0 to GPU 1
    CUDA_CHECK(cudaMemcpyPeer(d_dst, 1, d_src, 0, SIZE * sizeof(int)));


    // Verify data - copy GPU 1 back to host
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_src[i] != h_dst[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_src[i] << " != " << h_dst[i] << std::endl;
            break;
        }
    }

    std::cout << (success ? "Data transfer successful!" : "Data transfer failed!") << std::endl;

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    delete[] h_src;
    delete[] h_dst;

    return 0;
}