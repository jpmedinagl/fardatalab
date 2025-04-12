#include "ring.h"

RingBuffer::RingBuffer(void* buf, size_t num_chunks)
    : buffer(buf), size(num_chunks), count(0) {
    head = buf;
    tail = buf;
}

RingBufferRemoteInfo RingBuffer::export_metadata() const {
    return {
        buffer,
        &head,
        &tail,
        head,
        tail,
        size
    };
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
