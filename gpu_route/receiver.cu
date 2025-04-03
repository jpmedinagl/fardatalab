#include "utils.h"

void exchange_addresses(gpu_worker_t* local, int sockfd) 
{
    // Exchange worker addresses
    ucp_address_t* local_address;
    size_t local_address_length;
    UCS_CHECK(ucp_worker_get_address(local->worker, &local_address, &local_address_length));
    
    // Send/recv address length
    socket_recv(sockfd, &local_address_length, sizeof(local_address_length));
    
    // Send/recv address
    ucp_address_t* remote_address = (ucp_address_t*)malloc(local_address_length);
    socket_recv(sockfd, remote_address, local_address_length);
    
    // Create endpoint to remote worker
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = remote_address;
    UCS_CHECK(ucp_ep_create(local->worker, &ep_params, &local->ep));
    
    free(remote_address);
    
    ucp_worker_release_address(local->worker, local_address);
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
    close(sockfd);
    
    char* host_buf = (char*)malloc(BUFFER_SIZE);
    
    // Polling loop
    while (ucp_worker_progress(worker.worker)) {
        // 1. Check memory periodically
        CUDA_CHECK(cudaMemcpy(host_buf, worker.gpu_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost));
        
        // 2. Simple pattern verification
        if (strncmp(host_buf, "Hello", 5) == 0) {
            printf("Received data: %.*s\n", BUFFER_SIZE, host_buf);
            break;
        }
    }
    
    free(host_buf);

    return 0;
}