#ifndef UTILS_H
#define UTILS_H

#include <ucp/api/ucp.h>
#include <cuda_runtime.h>
#include <ucs/memory/memory_type.h>
#include <ucp/api/ucp_def.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <inttypes.h>

#include <stdio.h>

#define NUM_CHUNKS 10
#define PORT 12345

#define CHECK_ERROR(_cond, _msg)                                               \
    do {                                                                       \
        if (_cond) {                                                           \
            printf("Failed to %s\n", _msg);                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(_func)                                                      \
    do {                                                                       \
        cudaError_t _result = (_func);                                         \
        if (cudaSuccess != _result) {                                          \
            printf("%s failed: %s\n", #_func, cudaGetErrorString(_result));    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define UCS_CHECK(_expr)                                                       \
    do {                                                                       \
        ucs_status_t _status = (_expr);                                        \
        if (UCS_OK != _status) {                                               \
            printf("%s failed: %s\n", #_expr, ucs_status_string(_status));     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Basic socket send, loop to ensure full data transfer
void socket_send(int sockfd, const void* data, size_t size);

// Basic socket recv, loop to ensure full data transfer
void socket_recv(int sockfd, void* buffer, size_t size);

void init(ucp_context_h context, ucp_worker_h worker);

void create_ep(int sockfd, ucp_worker_h worker, ucp_ep_h * ep);

#endif // UTILS_H
