#include <cuda/atomic>
#include <cstdio>

__global__ void block_scope_example(cuda::atomic<int, cuda::thread_scope_block>* counter) {
    counter->fetch_add(1, cuda::memory_order_relaxed);
    __syncthreads(); // Ensure all threads in the block update before reading

    if (threadIdx.x == 0)
        printf("Block %d sees counter = %d\n", blockIdx.x, counter->load(cuda::memory_order_relaxed));
}

int main() {
    cuda::atomic<int, cuda::thread_scope_block>* d_counter;
    cudaMallocManaged(&d_counter, sizeof(cuda::atomic<int, cuda::thread_scope_block>));
    new (d_counter) cuda::atomic<int, cuda::thread_scope_block>(0); // Initialize

    block_scope_example<<<2, 4>>>(d_counter); // Two blocks of 4 threads each
    cudaDeviceSynchronize();

    return 0;
}

