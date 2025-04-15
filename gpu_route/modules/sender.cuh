#ifndef SENDER_H
#define SENDER_H

#include <ucp/api/ucp.h>
#include <cstddef>
#include <cstdint>

#include "ring.cuh"

class Sender {
private:
    ucp_context_h context;
    ucp_worker_h worker;

    // ring buffer is registered for rdma
    ucp_mem_h memh;
    RingBuffer * d_ringbuf;

    // needs to keep track of the next things for every single gpu...?
    ucp_ep_h ep;

    // ucp_address_t* remote_worker;
    ucp_rkey_h remote_rkey;

    void * remote_buf;
    // void ** remote_head_ptr;
    void ** remote_tail_ptr;
    void * remote_head;
    void * remote_tail;
    size_t size;

    void process_req(void * request);

    void recv_addr(int sockfd);

public:
    Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd);

    bool push(void * data, size_t size);

    void remote_push(int gpu_id);
};

#endif // SENDER_H