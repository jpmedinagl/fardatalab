#include "utils.h"

void get(gpu_worker_t* worker, void* data, size_t size)
{
    ucp_request_param_t params;
    memset(&params, 0, sizeof(params));
    params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.memory_type = UCS_MEMORY_TYPE_CUDA;

    void* request = ucp_get_nbx(worker->ep,
                               data,
                               size,
                               (uintptr_t)worker->remote_buffer_addr,
                               worker->remote_rkey,
                               &params);

    if (UCS_PTR_IS_ERR(request)) {
        printf("GET failed: %s\n", ucs_status_string(UCS_PTR_STATUS(request)));
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
    // Receiver GPU
    int gpu_id = 1;

    gpu_worker_t worker;
    init_gpu_worker(&worker, gpu_id);
    
    // Socket setup
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    listen(sockfd, 1);
    sockfd = accept(sockfd, NULL, NULL);
    
    exchange_addresses(&worker, sockfd);

    // gpu is ready to get data from other gpu    
    get(&worker, worker.gpu_buffer, BUFFER_SIZE);
    
    printf("\n");
    printf("GPU %d (receiver)\n", worker.gpu_id);
    
    char * host_buf = (char *)malloc(BUFFER_SIZE * sizeof(char));
    CUDA_CHECK(cudaMemcpy(host_buf, worker.gpu_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost));
    printf("Received: %.*s\n", BUFFER_SIZE, host_buf);
    
    free(host_buf);

    return 0;
}