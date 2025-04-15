#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstddef>
#include <cstdint>

#define CHUNK_SIZE 256

struct RingBufferRemoteInfo {
    void * buffer_addr;
    // void ** head_addr_ptr;
    void ** tail_addr_ptr;
    void * head_addr;
    void * tail_addr;
    size_t size;
};

class RingBuffer {
public:
    void * buffer;
    void * head;
    void * tail;
    size_t size;
    size_t count;

    // RingBuffer(void * buf, size_t num_chunks);

    __device__ void init(void* buf, size_t num_chunks) {
        buffer = buf;
        head = buf;
        tail = buf;
        size = num_chunks + 1;
        count = 0;
    }

    RingBufferRemoteInfo export_metadata() const;

    __device__ bool is_empty() const;
    __device__ bool is_full() const; 
    __device__ bool dequeue(void* out_chunk);
    __device__ bool enqueue(const void * chunk);
};

#endif // RING_BUFFER_H