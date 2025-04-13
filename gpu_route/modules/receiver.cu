#include "receiver.cuh"

#include <iostream>

__global__ void init_ringbuffer_kernel(RingBuffer* rb, void* buffer, size_t num_chunks) 
{
    new (rb) RingBuffer(buffer, num_chunks);
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   ucp_mem_h memh)
    : context(ctx), worker(wrk), ep(endpoint), memh(memh) 
{    
    // 1. allocate buffer 
    void* gpu_buffer;
    const size_t total_size = sizeof(RingBuffer) + (NUM_CHUNKS + CHUNK_SIZE);
    cudaMalloc(&gpu_memory, total_size);

    ucp_mem_map_params_t params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = gpu_memory,
        .length = total_size,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    ucp_mem_map(context, &params, &memh);

    // 3. intialize ring buffer
    d_ringbuf = reinterpret_cast<RingBuffer*>(gpu_memory);
    void* data_buffer = reinterpret_cast<char*>(gpu_memory) + sizeof(RingBuffer);

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