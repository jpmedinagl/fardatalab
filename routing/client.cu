#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#include "utils.h"

#define AM_ID_RESPONSE 0

static int connection_closed = 0;
static int response_received = 0;

/**
 * Initialize the worker for the client. We are using a single thread to 
 * process requests.
 */
int worker_init(ucp_context_h ucp_context, ucp_worker_h *ucp_worker)
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;
    int ret = 0;

    memset(&worker_params, 0, sizeof(worker_params));

    // worker uses single thread mode
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    // create worker using ucp
    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed ucp_worker_create\n");
        ret = -1;
    }
    return ret;
}


/**
 * Initializes the UCX client by setting up context and worker.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_init(ucx_client_t *client) 
{
    ucp_params_t ucp_params;
    ucs_status_t status;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));

    // feature types: TAG, STREAM, RMA, AM -> AM
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_AM;

    // initialize context using ucp
    status = ucp_init(&ucp_params, NULL, &(client->ucp_context));
    if (status != UCS_OK) {
        fprintf(stderr, "failed ucp_init\n");
        ret = -1;
        return ret;
    }

    // initalize worker 
    ret = worker_init(client->ucp_context, &(client->ucp_worker));
    if (ret != 0) {
        ucx_client_cleanup(client);
    }
    
    return ret;
}

/**
 * Callback for error. Specifically used for connecting to server.
 */
void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
    connection_closed = 1;
}

/**
 * Initializes a sockaddr_storage structure with the given IPv4 address and port.
 */
void set_sock_server_addr(const char *address_str, int port, struct sockaddr_storage *saddr)
{
    struct sockaddr_in * sa_in;

    memset(saddr, 0, sizeof(*saddr));

    sa_in = (struct sockaddr_in*)saddr;
    sa_in->sin_family = AF_INET;
    sa_in->sin_port = htons(port);

    if (inet_pton(AF_INET, address_str, &(sa_in->sin_addr)) <= 0) {
        fprintf(stderr, "Invalid address: %s\n", address_str);
        exit(EXIT_FAILURE);
    }

    printf("client connected to IP %s port %d\n", address_str, port);
}

/**
 * Connects the UCX client to a server at the given IP and port. Creates an endpoint.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_connect(ucx_client_t *client, const char *ip, int port)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_storage connect_addr;
    ucs_status_t status;

    set_sock_server_addr(ip, port, &connect_addr);

    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR   |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_cb;
    ep_params.err_handler.arg  = NULL;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);

    status = ucp_ep_create(client->ucp_worker, &ep_params, &client->ucp_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to connect to %s (%s)\n", ip,
                ucs_status_string(status));
        return -1;
    }

    return 0;
}

/**
 * Handles received Active Messages (AM) data. Part of the ucp callback. 
 * Allocates the total size from header, and copies full message.  
 */
ucs_status_t recv_am_data_cb(void *arg, const void *header, size_t header_length,
                               void *data, size_t length,
                               const ucp_am_recv_param_t *param)
{
    size_t total_data_received = 0;
    size_t total_data_size = 0;
    char *received_data = NULL;

    // The first time we enter the callback, we need to extract the total size
    if (total_data_size == 0) {
        // We expect the header to contain the total data size (just like in the server)
        if (header_length != sizeof(size_t)) {
            fprintf(stderr, "Received invalid header length %ld\n", header_length);
            return UCS_ERR_INVALID_PARAM;
        }
        
        total_data_size = *(size_t *)header;  // Get the total data size from the header

        // Allocate memory for the full data we're going to receive
        received_data = malloc(total_data_size + 1);
        if (!received_data) {
            fprintf(stderr, "Failed to allocate memory for received data\n");
            return UCS_ERR_NO_MEMORY;
        }
    }

    // Copy the incoming chunk of data to the allocated buffer
    memcpy(received_data + total_data_received, data, length);
    total_data_received += length;

    received_data[total_data_size] = '\0';

    // If we've received the full data, process it
    if (total_data_received >= total_data_size) {
        printf("Received data: %s (%ld)\n", received_data, total_data_size);

        // Clean up after receiving the full data
        free(received_data);
        received_data = NULL;
        total_data_received = 0;
        total_data_size = 0;

        // Mark the response as fully received (if you have such a flag)
        response_received = 1;
    }

    return UCS_OK;
}

/**
 * Registers a callback to handle incoming active messages on the worker.  
 * Associates the message ID with the callback function for processing received data.  
 */
ucs_status_t register_am_recv_callback(ucp_worker_h worker)
{
    ucp_am_handler_param_t param;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = AM_ID_RESPONSE;
    param.cb         = recv_am_data_cb;
    param.arg        = worker; 

    return ucp_worker_set_am_recv_handler(worker, &param);
}

/**
 * Callback function for handling the completion of an active messages send operation.  
 * Frees the request and logs an error if the send operation fails.  
 */
void send_callback(void *request, ucs_status_t status, void *user_data)
{
    if (status != UCS_OK) {
        fprintf(stderr, "Send failed: %s\n", ucs_status_string(status));
    }
    ucp_request_free(request);
}

int register_gpu_memory(ucp_context_h context, void *d_ptr, size_t size, ucp_mem_h *memh) {
    ucp_mem_map_params_t params;
    ucs_status_t status;

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address = d_ptr;
    params.length = size;
    params.flags = UCP_MEM_MAP_FIXED | UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(context, &params, memh);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to register GPU memory (%s)\n", ucs_status_string(status));
        return -1;
    }
    return 0;
}

/**
 * Sends an active message to the specified endpoint with a length-prefixed header.  
 * Uses send_callback to handle completion and returns the operation status.  
 */
ucs_status_t send_am_data(ucp_worker_h ucp_worker, ucp_ep_h ep, const char *msg, size_t msg_length)
{
    if (!ep) {
        fprintf(stderr, "send_am_data: Attempted to use a NULL endpoint!\n");
        return UCS_ERR_INVALID_PARAM;
    }

    ucp_request_param_t param;
    void *request;
    ucs_status_t status;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send      = send_callback;
    param.user_data    = NULL;
    param.memory_type = UCS_MEMORY_TYPE_CUDA;

    /* Create a header containing the message length */
    size_t header = msg_length;

    /* Send an Active Message (AM) to the server with the header */
    request = ucp_am_send_nbx(ep, AM_ID_RESPONSE, &header, sizeof(size_t), msg, msg_length, &param);

    if (request == NULL) {
        return UCS_OK;  // Operation completed immediately
    }

    if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
        fprintf(stderr, "Failed to send AM message: %s\n", ucs_status_string(status));
        return status;
    }

    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        ucp_worker_progress(ucp_worker);
    }

    status = ucp_request_check_status(request);

    return status;
}

/**
 * Sends a message to the server.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_send(ucx_client_t *client, const void *data, size_t length)
{
    ucs_status_t status;

    printf("\nSending data: %s (%ld)\n", (char *)data, length);

    // Send the request to the server
    status = send_am_data(client->ucp_worker, client->ucp_ep, data, length);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to send path to server.\n");
        return -1;
    }

    return 0;
}


/**
 * Registers the receive am callback. Once a request is received, it will use
 * callback function.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_receive(ucx_client_t *client)
{
    ucs_status_t status;

    // Register the receive callback for server responses
    status = register_am_recv_callback(client->ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to register AM receive callback.\n");
        return -1;
    }

    // Wait for a response from the server
    response_received = 0;  // Reset flag before waiting
    while (!response_received) {
        ucp_worker_progress(client->ucp_worker);
    }

    // retry limit - could be implemented in case server timeouts

    // int retries = 10;  // Set a retry limit (adjust as needed)

    // while (!response_received && retries > 0) {
    //     ucp_worker_progress(client->ucp_worker);
    //     usleep(100000);  // Sleep for 100ms to avoid CPU overuse
    //     retries--;
    // }

    // if (!response_received) {
    //     fprintf(stderr, "Receive timeout! Server may be down.\n");
    //     return -1;
    // }

    printf("\nReceived response from server.\n");
    return 0;
}

/**
 * Sends a request (path) to the server and waits for a response.  
 * Returns 0 on success, non-zero on failure.  
 */
int client_do_work(ucx_client_t *client, const char *path)
{
    int ret;

    // Send data
    ret = ucx_client_send(client, path, strlen(path));
    if (ret != 0) {
        return ret;
    }

    // Receive response
    ret = ucx_client_receive(client);
    if (ret != 0) {
        return ret;
    }

    return 0;
}

/**
 * Runs the UCX client by connecting to the server, sending a request, and receiving a response.  
 * Closes the connection once received.  
 */
int ucx_client_run(ucx_client_t * client, const char * server_addr, int port, char * path)
{
    ucs_status_t status;
    int ret;

    status = ucx_client_connect(client, server_addr, port);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to start client (%s)\n", ucs_status_string(status));
        return -1;
    }

    ret = client_do_work(client, path);

    /* Close the endpoint to the server */
    ep_close(client->ucp_worker, client->ucp_ep, UCP_EP_CLOSE_FLAG_FORCE);

    return ret;
}

/**
 * Cleans up UCX resources.
 */
void ucx_client_cleanup(ucx_client_t *client)
{
    if (client->ucp_worker) {
        ucp_worker_destroy(client->ucp_worker);
    }
    if (client->ucp_context) {
        ucp_cleanup(client->ucp_context);
    }
}

int main(int argc, char **argv)
{   
    char * path = NULL;
    char * server_addr = NULL;
    int port;
    int ret;

    ucx_client_t client;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server_addr> <path>\n", argv[0]);
        return -1;
    }

    server_addr = argv[1];
    port = DEFAULT_PORT;
    path = argv[2];

    CUDA_CHECK(cudaMalloc((void **)&d_path, strlen(path) + 1));
    CUDA_CHECK(cudaMemcpy(d_path, path, strlen(path), cudaMemcpyHostToDevice));
    char null_char = '\0';
    CUDA_CHECK(cudaMemcpy(d_path + strlen(path), &null_char, 1, cudaMemcpyHostToDevice));

    ret = ucx_client_init(&client);
    if (ret != 0) {
        fprintf(stderr, "failed to initilize client\n");
        return -1;
    }

    ret = register_gpu_memory(client.ucp_context, d_path, strlen(path) + 1, &client.ucp_memh);
    if (ret != 0) {
        ucx_client_cleanup(&client);
        cudaFree(d_path);
        return -1;
    }

    ret = ucx_client_run(&client, server_addr, port, path);
    ucx_client_cleanup(&client);

    cudaFree(d_path);

    return ret;
}