#include <ucp/api/ucp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// Set the size of the message (1024 bytes for this case)
#define MESSAGE_SIZE 1024

int main() {    
    // Initialize UCX for GPU-aware communication
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;

    // Create UCX context
    ucp_config_t *config;
    ucp_worker_params_t worker_params;

    // Initialize message data
    char message[MESSAGE_SIZE] = "This is a test message from GPU 0 to GPU 1.";
    size_t message_length = strlen(message) + 1;  // Include null terminator

    // Allocate buffers for each GPU
    char *d_message_gpu1, *d_message_gpu2;

    // Create UCX context
    ucp_config_read(NULL, NULL, &config);
    ucp_init(config, NULL, &ucp_context);

    // Create a UCX worker
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCP_THREAD_MODE_SINGLE;
    ucp_worker_create(ucp_context, &worker_params, &ucp_worker);

    // Set up memory buffers on GPU 0
    cudaSetDevice(0);
    cudaMalloc((void**)&d_message_gpu1, MESSAGE_SIZE);
    cudaMemcpy(d_message_gpu1, message, MESSAGE_SIZE, cudaMemcpyHostToDevice);

    // Set up memory buffer on GPU 1
    cudaSetDevice(1);
    cudaMalloc((void**)&d_message_gpu2, MESSAGE_SIZE);

    // Send data from GPU 0
    ucp_tag_send_info_t send_info;
    send_info.tag = 123;  // Example tag
    send_info.length = message_length;

    ucp_tag_send_nb(ucp_worker, d_message_gpu1, MESSAGE_SIZE, ucp_dt_make_contig(1), send_info.tag, NULL);

    // Receive data on GPU 1
    ucp_tag_recv_info_t recv_info;
    ucp_tag_recv_nb(ucp_worker, d_message_gpu2, MESSAGE_SIZE, ucp_dt_make_contig(1), 123, NULL);

    // Wait for completion or do other work with ucp_worker_progress()
    ucp_worker_progress(ucp_worker);

    // Print the received message on GPU 1 (for verification)
    char received_message[MESSAGE_SIZE];
    cudaMemcpy(received_message, d_message_gpu2, MESSAGE_SIZE, cudaMemcpyDeviceToHost);
    printf("Received message on GPU 1: %s\n", received_message);

    // Clean up
    cudaFree(d_message_gpu1);
    cudaFree(d_message_gpu2);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);
}
