#include "receiver.cuh"

#include <iostream>

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   ucp_mem_h memh, void* gpu_buffer, size_t num_chunks)
    : context(ctx), worker(wrk), ep(endpoint), memh(memh) 
{    
    cudaMalloc(&d_ringbuf, sizeof(RingBuffer));
    init_ringbuffer_kernel<<<1, 1>>>(d_ringbuf, gpu_buffer, num_chunks);
    cudaDeviceSynchronize();
}

void Receiver::dequeue(void* out_chunk) 
{
    char* d_tmp;
    cudaMalloc(&d_tmp, CHUNK_SIZE);
    
    bool success = false;
    dequeue_kernel<<<1, 1>>>(d_ringbuf, d_tmp, &success);

    char host_data[CHUNK_SIZE + 1] = {0};

    if (success) {
        cudaMemcpy(host_buffer, d_tmp, CHUNK_SIZE, cudaMemcpyDeviceToHost);
        
        host_buffer[CHUNK_SIZE] = '\0';
        std::cout << "Dequeued: " << host_buffer << std::endl;
    } else {
        std::cout << "Buffer empty" << std::endl;
    }
}

// GPU kernel wrapper
__global__ void dequeue_kernel(RingBuffer* rb, void* out_chunk, bool* success) 
{
    *success = rb->dequeue(out_chunk);
}