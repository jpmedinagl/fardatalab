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

RingBufferRemoteInfo RingBuffer::export_metadata() const {
    return {
        buffer,
        // &head,
        &tail,
        head,
        tail,
        size
    };
}

__device__ void RingBuffer::init(void* buf, size_t num_chunks) {
    buffer = buf;
    head = buf;
    tail = buf;
    size = num_chunks + 1;
    count = 0;
}

__device__ 
bool RingBuffer::is_empty() const {
    return count == 0;
}

__device__ 
bool RingBuffer::is_full() const {
    return count == size;
}

__device__ 
bool RingBuffer::enqueue(const void* data) {
    if (is_full()) {
        return false;
    }

    memcpy(tail, data, CHUNK_SIZE);

    uintptr_t offset = (reinterpret_cast<uintptr_t>(tail) - 
                        reinterpret_cast<uintptr_t>(buffer) + CHUNK_SIZE) % 
                        (size * CHUNK_SIZE);
    tail = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + offset);

    count--;
    return true;
}


__device__ 
bool RingBuffer::dequeue(void* out_chunk) {
    if (is_empty()) {
        return false;
    }

    memcpy(out_chunk, head, CHUNK_SIZE);

    uintptr_t offset = (reinterpret_cast<uintptr_t>(head) - 
                        reinterpret_cast<uintptr_t>(buffer) + CHUNK_SIZE) 
                        % (size * CHUNK_SIZE);
    head = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + offset);

    count--;
    return true;
}

#endif // RING_BUFFER_H