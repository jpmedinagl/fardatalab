#include "sender.cuh"

Sender::Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint)
    : context(ctx), worker(wrk), ep(endpoint), 
      remote_worker(nullptr), remote_rkey(nullptr),
      remote_head(nullptr), remote_tail(nullptr) 
{
    std::cout << "Sender object created." << std::endl;
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

void Sender::remote_push() 
{
    if (!remote_tail || !remote_head) {
        std::cerr << "Remote head or tail is null. Cannot enqueue data." << std::endl;
    }

    // 0. get the data we are sending from our local send buffer
    void* local_head;
    get_head_kernel<<<1,1>>>(d_ringbuf, &local_head);
    cudaDeviceSynchronize();

    // fetch the current head ? and check the count ?

    // 1. new tail position
    uintptr_t new_offset = ((uintptr_t)remote_tail - 
                            (uintptr_t)remote_buf + CHUNK_SIZE) %
                            (size * CHUNK_SIZE);
    
    void* new_tail = (void*)((uintptr_t)remote_buf + new_offset);

    // 2. update REMOTE tail pointer first
    ucp_request_param_t tail_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    
    void* tail_req = ucp_put_nbx(
        ep,
        &new_tail,
        sizeof(size_t),
        (uintptr_t)remote_tail_ptr,
        remote_rkey,
        &tail_params
    );
    process_request(tail_req);

    // 3. write data to old tail position
    ucp_request_param_t put_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };

    void* put_req = ucp_put_nbx(
        ep,
        local_head,
        CHUNK_SIZE,
        (uintptr_t)remote_tail,
        remote_rkey,
        &put_params
    );
    process_req(put_req);

    // 4. update local reference of the tail
    remote_tail = new_tail;
}

__global__ void push_kernel(RingBuffer* rb, void* data, size_t size, bool* success) {
    *success = rb->enqueue(data, size);
}

__global__ void get_head_kernel(RingBuffer* rb, void** head) {
    *head = rb->head;
}
