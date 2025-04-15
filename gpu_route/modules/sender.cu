#include "sender.cuh"

#include <new>

__global__ void init_ringbuffer_kernel(RingBuffer* rb, void* buffer, size_t num_chunks) 
{
    // new (rb) RingBuffer(buffer, num_chunks);
    rb->init(buffer, num_chunks);
}

__global__ void push_kernel(RingBuffer* rb, void* data, size_t size, bool* success) {
    *success = rb->enqueue(data); //, CHUNK_SIZE);
}

__global__ void get_head_kernel(RingBuffer* rb, void** head) {
    *head = rb->head;
}

void Sender::recv_addr(int sockfd)
{
    // 1. receive remote key
    size_t remote_rkey_size;
    socket_recv(sockfd, &remote_rkey_size, sizeof(remote_rkey_size));

    void *remote_rkey_buffer = malloc(remote_rkey_size);
    socket_recv(sockfd, remote_rkey_buffer, remote_rkey_size);

    printf("Rkey recv: %p %zu\n", remote_rkey_buffer, remote_rkey_size);

    UCS_CHECK(ucp_ep_rkey_unpack(ep, remote_rkey_buffer, &remote_rkey));

    free(remote_rkey_buffer);

    // 2. receive ring buffer information
    RingBufferRemoteInfo buf;
    socket_recv(sockfd, &buf, sizeof(buf));

    remote_buf = buf.buffer_addr;
    remote_tail_ptr = buf.tail_addr_ptr;
    remote_head = buf.head_addr;
    remote_tail = buf.tail_addr;
    size = buf.size;

    printf("\nRemote ring buffer info:\n");
    printf("buf: %p\n", remote_buf);
    printf("tail_ptr: %p\n", remote_tail_ptr);
    printf("head: %p\n", remote_head);
    printf("tail: %p\n", remote_tail);
    printf("size: %p\n", size);
}

Sender::Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{
    // sender doesn't need to register own ring buffer
    // 1. allocate ring buffer
    cudaMalloc(&this->d_ringbuf, sizeof(RingBuffer));
    
    void* data_buffer;
    cudaMalloc(&data_buffer, NUM_CHUNKS * CHUNK_SIZE);

    init_ringbuffer_kernel<<<1, 1>>>(this->d_ringbuf, data_buffer, NUM_CHUNKS);
    cudaDeviceSynchronize();

    recv_addr(sockfd);
}

void Sender::process_req(void* request) 
{
    if (UCS_PTR_IS_ERR(request)) {
        printf("UCX operation failed: %s\n", 
              ucs_status_string(UCS_PTR_STATUS(request)));
        exit(1);
    }
    
    ucs_status_t status;
    do {
        ucp_worker_progress(worker);
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
    
    ucp_request_free(request);
}

bool Sender::push(void* data, size_t size) {
    bool success;
    push_kernel<<<1,1>>>(d_ringbuf, data, size, &success);
    cudaDeviceSynchronize();
    return success;
}

void Sender::remote_push(int gpu_id) 
{   
    (void *) gpu_id;
    if (!remote_tail || !remote_head) {
        printf("Remote head or tail is null. Cannot enqueue data.\n");
    }

    // 0. get the data we are sending from our local send buffer
    void* local_head;
    get_head_kernel<<<1,1>>>(d_ringbuf, &local_head);
    cudaDeviceSynchronize();

    char* host_chunk = (char*)malloc(CHUNK_SIZE);
    cudaMemcpy(host_chunk, local_head, CHUNK_SIZE, cudaMemcpyDeviceToHost);
    printf("Chunk contents (first 64 bytes):\n");
    for (int i = 0; i < std::min(CHUNK_SIZE, 64); ++i) {
        printf("%c", host_chunk[i]);
    }
    printf("\n");
    free(host_chunk);

    // fetch the current head ? and check the count ?

    // 1. new tail position
    uintptr_t new_offset = (remote_tail - remote_buf + CHUNK_SIZE) %
                            (size * CHUNK_SIZE);
    
    void* new_tail = (void*)(remote_buf + new_offset);

    printf("old tail: %p\n", remote_tail);
    printf("new tail: %p\n", new_tail);
    printf("updated tail ptr: %p\n", remote_tail_ptr);

    // 2. update REMOTE tail pointer first
    ucp_request_param_t tail_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    
    void* tail_req = ucp_put_nbx(
        ep,
        &new_tail,
        sizeof(size_t),
        remote_tail_ptr,
        remote_rkey,
        &tail_params
    );
    process_req(tail_req);

    printf("tail written\n");

    // 3. write data to old tail position
    ucp_request_param_t put_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };

    void* put_req = ucp_put_nbx(
        ep,
        local_head,
        CHUNK_SIZE,
        remote_tail,
        remote_rkey,
        &put_params
    );
    process_req(put_req);

    printf("data placed\n");

    // 4. update local reference of the tail
    remote_tail = (uintptr_t)new_tail;
}
