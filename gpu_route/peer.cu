#include "utils.h"
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdio>

int init_gpu_worker(gpu_worker_t* worker, int gpu_id) 
{   
    // Initialize worker and gpu buffer + register memory
    CUDA_CHECK(cudaSetDevice(gpu_id));

    CUDA_CHECK(cudaMalloc(&worker->gpu_buffer, BUFFER_SIZE));
    worker->buffer_size = BUFFER_SIZE;

    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA;

    UCS_CHECK(ucp_init(&params, NULL, &worker->context));

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    UCS_CHECK(ucp_worker_create(worker->context, &worker_params, &worker->worker));

    ucp_mem_map_params_t mem_params;
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                          UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.address = worker->gpu_buffer;
    mem_params.length = worker->buffer_size;
    mem_params.flags = UCP_MEM_MAP_FIXED;

    UCS_CHECK(ucp_mem_map(worker->context, &mem_params, &worker->memh));
    
    return 0;
}

void socket_send(int sockfd, const void* data, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        ssize_t res = send(sockfd, (char*)data + sent, size - sent, 0);
        if (res <= 0) {
            perror("send failed");
            exit(1);
        }
        sent += res;
    }
}

void socket_recv(int sockfd, void* buffer, size_t size) {
    size_t received = 0;
    while (received < size) {
        ssize_t res = recv(sockfd, (char*)buffer + received, size - received, 0);
        if (res <= 0) {
            perror("recv failed");
            exit(1);
        }
        received += res;
    }
}

void exchange_addresses(gpu_worker_t* local, int sockfd, int is_initiator) 
{
    // Exchange worker addresses
    ucp_address_t* local_address;
    size_t local_address_length;
    UCS_CHECK(ucp_worker_get_address(local->worker, &local_address, &local_address_length));
    
    // Send/recv address length
    if (is_initiator) {
        socket_send(sockfd, &local_address_length, sizeof(local_address_length));
    } else {
        socket_recv(sockfd, &local_address_length, sizeof(local_address_length));
    }
    
    // Send/recv address
    if (is_initiator) {
        socket_send(sockfd, local_address, local_address_length);
    } else {
        ucp_address_t* remote_address = (ucp_address_t*)malloc(local_address_length);
        socket_recv(sockfd, remote_address, local_address_length);
        
        // Create endpoint to remote worker
        ucp_ep_params_t ep_params;
        memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address = remote_address;
        UCS_CHECK(ucp_ep_create(local->worker, &ep_params, &local->ep));
        
        free(remote_address);
    }
    
    ucp_worker_release_address(local->worker, local_address);
}

void perform_put(gpu_worker_t* worker, void* data, size_t size) {
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

void run_peer(int gpu_id, const char* peer_ip, int is_initiator) {
    gpu_worker_t worker;
    init_gpu_worker(&worker, gpu_id);
    
    // Socket setup
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    
    if (is_initiator) {
        addr.sin_addr.s_addr = inet_addr(peer_ip);
        if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr))) {
            perror("connect failed");
            exit(1);
        }
    } else {
        addr.sin_addr.s_addr = INADDR_ANY;
        bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
        listen(sockfd, 1);
        sockfd = accept(sockfd, NULL, NULL);
    }
    
    exchange_addresses(&worker, sockfd, is_initiator);
    close(sockfd);
    
    if (is_initiator) {
        // Send data
        char init_data[BUFFER_SIZE] = "Hello from GPU!";
        CUDA_CHECK(cudaMemcpy(worker.gpu_buffer, init_data, BUFFER_SIZE, cudaMemcpyHostToDevice));
        perform_put(&worker, worker.gpu_buffer, BUFFER_SIZE);
    }
    
    // Progress loop
    while (ucp_worker_progress(ucp_worker)) {
        // Process events until all operations are completed
    }
}
