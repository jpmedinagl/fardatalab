#include "peer.cu"

void exchange_addresses(gpu_worker_t* local, int sockfd) 
{
    // Exchange worker addresses
    ucp_address_t* local_address;
    size_t local_address_length;
    UCS_CHECK(ucp_worker_get_address(local->worker, &local_address, &local_address_length));
    
    // Send/recv address length
    socket_send(sockfd, &local_address_length, sizeof(local_address_length));
    
    // Send/recv address
    socket_send(sockfd, local_address, local_address_length);    
    
    ucp_worker_release_address(local->worker, local_address);
}

void perform_put(gpu_worker_t* worker, void* data, size_t size) 
{
    ucp_request_param_t params;
    memset(&params, 0, sizeof(params));
    params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.memory_type = UCS_MEMORY_TYPE_CUDA;
    
    // In real code, you'd exchange rkey and remote address first
    void* request = ucp_put_nbx(worker->ep,
                               data,
                               size,
                               (uintptr_t)worker->gpu_buffer, // Remote address
                               NULL, // Would be rkey in real code
                               &params);
    
    if (UCS_PTR_IS_PTR(request)) {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker->worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
    } else if (UCS_PTR_IS_ERR(request)) {
        printf("PUT failed: %s\n", ucs_status_string(UCS_PTR_STATUS(request)));
    }
}

int main() 
{
    // Sender GPU
    int gpu_id = 0;
    char ip = "192.168.1.2";

    gpu_worker_t worker;
    init_gpu_worker(&worker, gpu_id);
    
    // Socket setup
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    
    addr.sin_addr.s_addr = inet_addr(peer_ip);
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr))) {
        perror("connect failed");
        exit(1);
    }
    
    exchange_addresses(&worker, sockfd);
    close(sockfd);
    
    // Send data
    char init_data[BUFFER_SIZE] = "Hello from GPU!";
    CUDA_CHECK(cudaMemcpy(worker.gpu_buffer, init_data, BUFFER_SIZE, cudaMemcpyHostToDevice));
    perform_put(&worker, worker.gpu_buffer, BUFFER_SIZE);
    
    
    // Progress loop
    while (ucp_worker_progress(ucp_worker)) {
        // Process events until all operations are completed
    }

    return 0;
}