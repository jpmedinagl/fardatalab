#ifndef SENDER_H
#define SENDER_H

#include <ucp/api/ucp.h>
#include <cstddef>
#include <cstdint>

class Sender {
private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;

    ucp_address_t* remote_worker;
    ucp_rkey_h remote_rkey;

    void * remote_buf;
    void ** remote_head_ptr;
    void * remote_head;
    void ** remote_tail_ptr;
    void * remote_tail;
    size_t size;

    void process_req(void * request);

public:
    Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
    ucp_mem_h memh, ucp_address_t* remote_worker, ucp_rkey_h remote_rkey);

    void enqueue_remote(void* data, size_t size);
};

#endif // SENDER_H