#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstddef>
#include <cstdint>

#define CHUNK_SIZE 256

struct RingBufferRemoteInfo {
    uintptr_t buffer_addr;
    void ** head_addr_ptr;
    void ** tail_addr_ptr;
    void * head_addr;
    void * tail_addr;
    size_t size;
};

class RingBuffer {
private:
    void * buffer;
    void * head;
    void * tail;
    size_t size;
    size_t count;

public:
    RingBuffer(void * buf, size_t num_chunks);

    RingBufferRemoteInfo export_metadata() const;

    __device__ bool is_empty() const;
    __device__ bool is_full() const; 
    __device__ bool dequeue(void* out_chunk);
    __device__ bool enqueue(const void * chunk);
};

#endif // RING_BUFFER_H