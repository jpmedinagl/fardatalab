#include "receiver.cuh"

#include <iostream>
#include <new>

__global__ void init_ringbuffer_kernel(RingBuffer* rb, void* buffer, size_t num_chunks) 
{
    // new (rb) RingBuffer(buffer, num_chunks);
    rb->init(buffer, num_chunks);
}

__global__ void dequeue_kernel(RingBuffer* rb, void* out_chunk, bool* success) 
{
    *success = rb->dequeue(out_chunk);
}

void Receiver::send_addr(int sockfd)
{
    // 1. send key
    size_t rkey_size;
    void *rkey_buffer;
    UCS_CHECK(ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size));
    
    printf("Rkey send: %p %zu\n", rkey_buffer, rkey_size);

    socket_send(sockfd, &rkey_size, sizeof(rkey_size));
    socket_send(sockfd, rkey_buffer, rkey_size);
    UCS_CHECK(ucp_rkey_buffer_release(rkey_buffer));

    // 2. send ring buffer information
    RingBufferRemoteInfo buf = d_ringbuf->export_metadata();
    socket_send(sockfd, &buf, sizeof(buf));        
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{    
    // 1. allocate buffer
    void* gpu_memory;
    const size_t total_size = sizeof(RingBuffer) + (NUM_CHUNKS + CHUNK_SIZE);
    cudaMalloc(&gpu_memory, total_size);

    // 2. map memory
    ucp_mem_map_params_t params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = gpu_memory,
        .length = total_size,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    UCS_CHECK(ucp_mem_map(context, &params, &memh));

    // 3. intialize ring buffer
    d_ringbuf = reinterpret_cast<RingBuffer*>(gpu_memory);
    void* data_buffer = reinterpret_cast<char*>(gpu_memory) + sizeof(RingBuffer);

    init_ringbuffer_kernel<<<1, 1>>>(d_ringbuf, data_buffer, NUM_CHUNKS);
    cudaDeviceSynchronize();

    send_addr(sockfd);
}

void Receiver::dequeue(void* out_chunk) 
{
    // dequeue from ring buffer
    bool success = false;
    dequeue_kernel<<<1, 1>>>(d_ringbuf, out_chunk, &success);

    if (!success) {
        std::cout << "Buffer empty" << std::endl;
    }
}
