#include <cuda_runtime.h>
#include <iostream>

#define SIZE 1024
#define BATCH 256

void checkCuda(cudaError_t err, const char * msg) 
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
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
    checkCuda(cudaSetDevice(0), "Set GPU 0");
    checkCuda(cudaMalloc((void**)&d_src, SIZE * sizeof(int)), "Allocate GPU 0 memory");

    // Copy data to GPU 0
    checkCuda(cudaMemcpy(d_src, h_src, SIZE * sizeof(int), cudaMemcpyHostToDevice), "Memcpy to GPU 0");

    // Set up GPU 1
    checkCuda(cudaSetDevice(1), "Set GPU 1");
    checkCuda(cudaMalloc((void **)&d_dst, SIZE * sizeof(int)), "Allocate GPU 1 memory");

    
    // Transfer GPU 0 to GPU 1
    checkCuda(cudaMemcpyPeer(d_dst, 1, d_src, 0, SIZE * sizeof(int)), "Memcpy Peer 0 to 1");


    // Verify data - copy GPU 1 back to host
    checkCuda(cudaMemcpy(h_dst, d_dst, SIZE * sizeof(int), cudaMemcpyDeviceToHost), "Memcpy back to host");

    bool success = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_src[i] != h_dst[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_src[i] << " != " << h_dst[i] << std::endl;
            break;
        }
    }

    std::cout << (success ? "Data transfer successful!" : "Data transfer failed!") << std::endl;

    checkCuda(cudaFree(d_src), "Free GPU 0 memory");
    checkCuda(cudaFree(d_dst), "Free GPU 1 memory");
    delete[] h_src;
    delete[] h_dst;

    return 0;
}