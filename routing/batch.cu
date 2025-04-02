#include <cuda_runtime.h>
#include <iostream>

#define SIZE 1024  // Total number of elements
#define BATCH_SIZE 256  // Number of elements per batch

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

int main() {
    int *d_src, *d_dst;
    int *h_src, *h_dst;

    // Allocate host memory
    h_src = new int[SIZE];
    h_dst = new int[SIZE];

    // Initialize host data
    for (int i = 0; i < SIZE; i++) {
        h_src[i] = i;
    }

    // Set device 0 and allocate memory
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&d_src, SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Set device 1 and allocate memory
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, SIZE * sizeof(int)));

    // Perform batched memory transfers
    for (int offset = 0; offset < SIZE; offset += BATCH_SIZE) {
        int batch_size = min(BATCH_SIZE, SIZE - offset);  // Handle last batch

        CUDA_CHECK(cudaMemcpyPeer(
            d_dst + offset, 1,  // Destination GPU (1)
            d_src + offset, 0,  // Source GPU (0)
            batch_size * sizeof(int)
        ), "Memcpy Peer (Batched)");
    }

    // Copy data from GPU 1 to host for verification
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify data
    bool success = true;
    for (int i = 0; i < SIZE; i++) {
        if (h_src[i] != h_dst[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": " << h_src[i] << " != " << h_dst[i] << std::endl;
            break;
        }
    }

    std::cout << (success ? "Batched Data transfer successful!" : "Data transfer failed!") << std::endl;

    // Free memory
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    delete[] h_src;
    delete[] h_dst;

    return 0;
}
