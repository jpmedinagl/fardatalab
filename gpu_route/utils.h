#include <ucp/api/ucp.h>
#include <cuda_runtime.h>
#include <ucs/memory/memory_type.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>     // For close()
#include <stdlib.h>     // For exit()
#include <string.h>     // For memset()
#include <cstdio>       // For perror()

#include <stdio.h>

#define BUFFER_SIZE 2048
#define PORT 12345

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

typedef struct {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;
    void* gpu_buffer;
    int buffer_size;
    int gpu_id;
} gpu_worker_t;
