#include "utils.h"

void put(gpu_worker_t* worker, void* data, size_t size) 
{
    ucp_request_param_t params;
    memset(&params, 0, sizeof(params));
    params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.memory_type = UCS_MEMORY_TYPE_CUDA;
    
    void* request = ucp_put_nbx(worker->ep,
                               data,
                               size,
                               (uintptr_t)worker->remote_buffer_addr,
                               worker->remote_rkey,
                               &params);

    if (UCS_PTR_IS_ERR(request)) {
        printf("PUT failed: %s\n", ucs_status_string(UCS_PTR_STATUS(request)));
        exit(1);
    }    

    ucs_status_t status;
    do {
        ucp_worker_progress(worker->worker);
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
    ucp_request_free(request);
}

int main() 
{
    // Sender GPU - ip ? local host ?
    int gpu_id = 0;
    char peer_ip[] = "127.0.0.1";

    gpu_worker_t worker;
    init_gpu_worker(&worker, gpu_id);

    // Put random data on GPU
    char data[BUFFER_SIZE];
    memset(data, 0, BUFFER_SIZE);
    strncpy(data, "Hello from GPU!", BUFFER_SIZE - 1);
    CUDA_CHECK(cudaMemcpy(worker.gpu_buffer, data, BUFFER_SIZE, cudaMemcpyHostToDevice));
    
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

    // CUDA_CHECK(cudaMalloc(&worker.gpu_buffer, BUFFER_SIZE));
    // printf("GPU buffer allocated at %p\n", worker.gpu_buffer);

    // sleep(0.5);

    put(&worker, worker.gpu_buffer, BUFFER_SIZE);
    
    printf("\n");
    printf("GPU %d (sender)\n", worker.gpu_id);
    printf("Sent: %.*s\n", BUFFER_SIZE, data);

    gpu_worker_teardown(&worker);

    return 0;
}