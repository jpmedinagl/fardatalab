#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <cstddef>
#include <cstdint>

#define CHUNK_SIZE 256

struct RingBufferRemoteInfo {
    uintptr_t buffer_addr;
    // void ** head_addr_ptr;
    uintptr_t tail_addr_ptr;
    uintptr_t head_addr;
    uintptr_t tail_addr;
    size_t size;
};

RingBufferRemoteInfo export_metadata(const RingBuffer& rb) {
    return {
        reinterpret_cast<uintptr_t>(rb.buffer),
        // &head,
        reinterpret_cast<uintptr_t>(&rb.tail),
        reinterpret_cast<uintptr_t>(rb.head),
        reinterpret_cast<uintptr_t>(rb.tail),
        rb.size
    };
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
    };

    __device__ bool is_empty() const {
        return count == 0;
    };

    __device__ bool is_full() const {
        return count == size;
    };

    __device__ bool enqueue(const void * chunk) {
        if (is_full()) {
            return false;
        }

        memcpy(tail, chunk, CHUNK_SIZE);

        uintptr_t offset = (reinterpret_cast<uintptr_t>(tail) - 
                            reinterpret_cast<uintptr_t>(buffer) + CHUNK_SIZE) % 
                            (size * CHUNK_SIZE);
        tail = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(buffer) + offset);

        count--;
        return true;
    };

    __device__ bool dequeue(void* out_chunk) {
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
    };
};

#endif // RING_BUFFER_H