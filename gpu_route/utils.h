#include <ucp/api/ucp.h>
#include <cuda_runtime.h>
#include <ucs/memory/memory_type.h>
#include <ucp/api/ucp_def.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>     // For close()
#include <stdlib.h>     // For exit()
#include <string.h>     // For memset()
#include <cstdio>       // For perror()
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <stdio.h>

#define BUFFER_SIZE 2048
#define PORT 12345

#ifndef UCP_FEATURE_CUDA
#define UCP_FEATURE_CUDA UCS_BIT(12)
#endif

#define CHECK_ERROR(_cond, _msg)                                               \
  do {                                                                         \
    if (_cond) {                                                               \
      printf("Failed to %s\n", _msg);                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(_func)                                                      \
  do {                                                                         \
    cudaError_t _result = (_func);                                             \
    if (cudaSuccess != _result) {                                              \
      printf("%s failed: %s\n", #_func, cudaGetErrorString(_result));         \
    }                                                                          \
  } while (0)

#define UCS_CHECK(_expr) \
    do { \
        ucs_status_t _status = (_expr); \
        if (UCS_OK != _status) { \
            printf("%s failed: %s\n", #_expr, ucs_status_string(_status)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

typedef struct {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;
    void* gpu_buffer;
    int buffer_size;
    int gpu_id;
} gpu_worker_t;

typedef struct {
    int received;
    void* buffer;
} packet_t;

int init_gpu_worker(gpu_worker_t* worker, int gpu_id) 
{   
    // Initialize worker and gpu buffer + register memory
    CUDA_CHECK(cudaSetDevice(gpu_id));

    CUDA_CHECK(cudaMalloc(&worker->gpu_buffer, BUFFER_SIZE));
    worker->buffer_size = BUFFER_SIZE;

    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_CUDA;
    params.estimated_num_eps = 1;

    UCS_CHECK(ucp_init(&params, NULL, &worker->context));

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    UCS_CHECK(ucp_worker_create(worker->context, &worker_params, &worker->worker));

    // ucp_mem_map_params_t mem_params;
    // memset(&mem_params, 0, sizeof(mem_params));
    // mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
    //                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
    //                       UCP_MEM_MAP_PARAM_FIELD_FLAGS |
    //                       UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    // mem_params.address = worker->gpu_buffer;
    // mem_params.length = worker->buffer_size;
    // mem_params.flags = UCP_MEM_MAP_ALLOCATE;
    // mem_params.memory_type = UCS_MEMORY_TYPE_CUDA;

    ucp_mem_map_params_t mem_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                    UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                    UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                    UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = worker->gpu_buffer,
        .length = worker->buffer_size,
        .flags = UCP_MEM_MAP_FIXED,  // For pre-allocated GPU memory
        .memory_type = UCS_MEMORY_TYPE_CUDA  // Explicit CUDA memory type
    };

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
