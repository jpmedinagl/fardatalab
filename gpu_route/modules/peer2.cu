#include "utils.h"
#include "receiver.cuh"

int main() 
{
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;

    CUDA_CHECK(cudaSetDevice(1));

    init(&context, &worker);
    
    // Socket setup
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;
    
    int optval = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    listen(sockfd, 1);
    sockfd = accept(sockfd, NULL, NULL);

    // exchange addresses + keys!

    printf("socked connected\n");

    create_ep(sockfd, worker, &ep);

    Receiver receiver(context, worker, ep, sockfd);

    printf("Receiver connected\n\n");

    sleep(5);

    for (int i = 0; i < 1; i++) {
        void * out_chunk;
        cudaMalloc(&out_chunk, CHUNK_SIZE);

        receiver.dequeue(out_chunk);

        char host_data[CHUNK_SIZE + 1] = {0};
        cudaMemcpy(host_data, out_chunk, CHUNK_SIZE, cudaMemcpyDeviceToHost);

        printf("Received chunk %d: %s\n", i, host_data);

        cudaFree(out_chunk);
    }

    return 0;
}