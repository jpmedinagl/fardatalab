#include "utils.h"

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

// Basic socket recv, loop to ensure full data transfer
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

void init(ucp_context_h * context, ucp_worker_h * worker)
{
    ucp_params_t params = {
        .field_mask = UCP_PARAM_FIELD_FEATURES,
        .features = UCP_FEATURE_RMA
    };
    
    UCS_CHECK(ucp_init(&params, NULL, context));

    ucp_worker_params_t worker_params = {
        .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCS_THREAD_MODE_SINGLE
    };

    UCS_CHECK(ucp_worker_create(*context, &worker_params, worker));
}

void create_ep(int sockfd, ucp_worker_h worker, ucp_ep_h * ep)
{   
    printf("creating ep...\n");

    ucp_address_t* local_worker_addr;
    size_t local_worker_len;
    UCS_CHECK(ucp_worker_get_address(worker, &local_worker_addr, &local_worker_len));
    
    uint64_t addr_header = *((uint64_t*)local_worker_addr);
    printf("Worker address: %p (%zu)\n", addr_header, local_worker_len);

    // exit(1);
    
    // Send to remote worker
    socket_send(sockfd, &local_worker_len, sizeof(local_worker_len));
    socket_send(sockfd, local_worker_addr, local_worker_len);

    ucp_worker_release_address(worker, local_worker_addr);

    ucp_address_t* remote_worker_addr;
    size_t remote_worker_len;
    
    socket_recv(sockfd, &remote_worker_len, sizeof(remote_worker_len));
    remote_worker_addr = (ucp_address_t*)malloc(remote_worker_len);
    socket_recv(sockfd, remote_worker_addr, remote_worker_len);

    uint64_t remote_addr_header = *((uint64_t*)remote_worker_addr);
    printf("Remote worker address: %p (%zu)\n", remote_addr_header, remote_worker_len);

    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = remote_worker_addr;

    UCS_CHECK(ucp_ep_create(worker, &ep_params, ep));

    printf("Endpoint created.\n");
}
