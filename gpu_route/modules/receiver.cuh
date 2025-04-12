#ifndef RECEIVER_H
#define RECEIVER_H

class Receiver {
private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;

    // add ring buffers for different gpus
    RingBuffer * d_ringbuf;
    // RingBuffer & buffer1; ...

public:
    Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
    ucp_mem_h memh, ucp_address_t* remote_worker, ucp_rkey_h remote_rkey);

    void dequeue(void * out_chunk);
};

#endif // RECEIVER_H