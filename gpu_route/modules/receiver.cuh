#ifndef RECEIVER_H
#define RECEIVER_H

#include "utils.h"
#include "ring.cuh"

class Receiver {
private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;


    // temporary debugging
    ucp_rkey_h remote_rkey;

    // add ring buffers for different gpus
    RingBuffer * d_ringbuf;
    // RingBuffer & buffer1; ...

    void send_addr(int sockfd);
    void recv_addr(int sockfd);

public:
    Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd);

    void dequeue(void * out_chunk);
};

#endif // RECEIVER_H