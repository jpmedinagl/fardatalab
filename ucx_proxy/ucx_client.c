#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#include "ucx_client.h"
#include "ucx_util.h"


#define AM_ID_RESPONSE 0

static int connection_closed = 0;
static int response_received = 0;

/**
 * Initialize the worker for this server. We are using a single thread to 
 * process incoming requests.
 */
int init_worker(ucp_context_h ucp_context, ucp_worker_h *ucp_worker)
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
        fprintf(stderr, "failed ucp_init");
        ret = -1;
        return ret;
    }

    // initalize worker 
    ret = init_worker(client->ucp_context, &(client->ucp_worker));
    if (ret != 0) {
        ucx_client_cleanup(client);
    }
    
    return ret;
}

/**
 * Connects the UCX client to a server at the given IP and port.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_connect(ucx_client_t *client, const char *ip, int port);

/**
 * Sends a message to the server.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_send(ucx_client_t *client, const void *data, size_t length);

/**
 * Receives a message from the server.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_receive(ucx_client_t *client, void *buffer, size_t length);

void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
    connection_closed = 1;
}

void set_sock_server_addr(const char *address_str, struct sockaddr_storage *saddr)
{
    struct sockaddr_in * sa_in;

    memset(saddr, 0, sizeof(*saddr));

    sa_in = (struct sockaddr_in*)saddr;
    // sa_in->sin_addr.s_addr = address_str;
    sa_in->sin_family = AF_INET;
    sa_in->sin_port = htons(DEFAULT_PORT);

    if (inet_pton(AF_INET, address_str, &(sa_in->sin_addr)) <= 0) {
        fprintf(stderr, "Invalid address: %s\n", address_str);
        exit(EXIT_FAILURE);
    }
}

ucs_status_t start_client(ucp_worker_h ucp_worker, const char *address_str, ucp_ep_h *client_ep)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_storage connect_addr;
    ucs_status_t status;

    set_sock_server_addr(address_str, &connect_addr);

    /*
     * Endpoint field mask bits:
     * UCP_EP_PARAM_FIELD_FLAGS             - Use the value of the 'flags' field.
     * UCP_EP_PARAM_FIELD_SOCK_ADDR         - Use a remote sockaddr to connect
     *                                        to the remote peer.
     * UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE - Error handling mode - this flag
     *                                        is temporarily required since the
     *                                        endpoint will be closed with
     *                                        UCP_EP_CLOSE_MODE_FORCE which
     *                                        requires this mode.
     *                                        Once UCP_EP_CLOSE_MODE_FORCE is
     *                                        removed, the error handling mode
     *                                        will be removed.
     */
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

    status = ucp_ep_create(ucp_worker, &ep_params, client_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to connect to %s (%s)\n", address_str,
                ucs_status_string(status));
    }

    return status;
}

ucs_status_t client_am_data_cb(void *arg, const void *header, size_t header_length,
                               void *data, size_t length,
                               const ucp_am_recv_param_t *param)
{
    static size_t total_data_received = 0;
    static size_t total_data_size = 0;  
    static char *received_data = NULL;  

    /* 🔴 FIX: Allocate `received_data` if it's NULL */
    if (received_data == NULL) {
        total_data_size = length;
        received_data = malloc(total_data_size);
        if (!received_data) {
            fprintf(stderr, "client_am_data_cb: Failed to allocate memory for received data\n");
            return UCS_ERR_NO_MEMORY;
        }
    }

    // Store received chunk
    memcpy(received_data + total_data_received, data, length);
    total_data_received += length;

    // If all data is received, process it
    if (total_data_received >= total_data_size) {
        printf("Client received full data: %s\n", received_data);
        free(received_data);
        received_data = NULL;
        total_data_received = 0;
        total_data_size = 0;

        /* 🔴 FIX: Mark the response as received */
        response_received = 1;
    }

    return UCS_OK;
}

ucs_status_t register_client_am_recv_callback(ucp_worker_h worker)
{
    ucp_am_handler_param_t param;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = AM_ID_RESPONSE;  // Different from request AM_ID
    param.cb         = client_am_data_cb;  // Client's receive handler
    param.arg        = worker; 

    return ucp_worker_set_am_recv_handler(worker, &param);
}

void send_callback(void *request, ucs_status_t status, void *user_data)
{
    if (status != UCS_OK) {
        fprintf(stderr, "Send failed: %s\n", ucs_status_string(status));
    }
    ucp_request_free(request);
}

ucs_status_t send_am_message(ucp_worker_h ucp_worker, ucp_ep_h ep, const char *msg, size_t msg_length)
{
    if (!ep) {
        fprintf(stderr, "send_am_message: Attempted to use a NULL endpoint!\n");
        return UCS_ERR_INVALID_PARAM;
    }

    printf("send_am_message\n");
    printf("%s, %ld\n", msg, msg_length);

    ucp_request_param_t param;
    void *request;
    ucs_status_t status;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send      = send_callback;
    param.user_data    = NULL;

    /* Create a header containing the message length */
    size_t header = msg_length;

    /* Send an Active Message (AM) to the server with the header */
    request = ucp_am_send_nbx(ep, AM_ID_RESPONSE, &header, sizeof(size_t), msg, msg_length, &param);

    /* 🔴 FIX: If the request completes immediately, return */
    if (request == NULL) {
        return UCS_OK;  // Operation completed immediately
    }

    if (UCS_PTR_IS_ERR(request)) {
        status = UCS_PTR_STATUS(request);
        fprintf(stderr, "Failed to send AM message: %s\n", ucs_status_string(status));
        return status;
    }

    /* 🔴 FIX: Wait for completion and ensure request is freed */
    while (ucp_request_check_status(request) == UCS_INPROGRESS) {
        ucp_worker_progress(ucp_worker);
    }

    status = ucp_request_check_status(request);

    /* 🔴 FIX: Always free the request */
    // ucp_request_free(request);

    return status;
}

int client_do_work(ucp_worker_h ucp_worker, ucp_ep_h ep, const char *path)
{
    int ret = 0;
    ucs_status_t status;

    // Register the receive callback for server responses
    status = register_client_am_recv_callback(ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to register AM receive callback.\n");
        ret = -1;
    }

    // Send the request to the server
    status = send_am_message(ucp_worker, ep, path, strlen(path));
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to send path to server.\n");
        ret = -1;
    }

    printf("Sent path to server: %s\n", path);

    // Wait for a response (if the server sends one)
    while (!response_received) {
        ucp_worker_progress(ucp_worker);
    }

    printf("Received response from server.\n");

// out:
    return ret;
}

int ucx_client_run(ucp_worker_h ucp_worker, const char * server_addr, char * path)
{
    ucp_ep_h client_ep;
    ucs_status_t status;
    int ret;

    status = start_client(ucp_worker, server_addr, &client_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to start client (%s)\n", ucs_status_string(status));
        // ret = -1;
        // goto out;
        return -1;
    }

    ret = client_do_work(ucp_worker, client_ep, path);

    /* Close the endpoint to the server */
    ep_close(ucp_worker, client_ep, UCP_EP_CLOSE_FLAG_FORCE);

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
    int ret;

    ucx_client_t client;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <server_addr> <path>\n", argv[0]);
        return -1;
    }

    server_addr = argv[1];
    path = argv[2];

    ret = ucx_client_init(&client);

    ret = ucx_client_run(client.ucp_worker, server_addr, path);

    ucp_worker_destroy(client.ucp_worker);
    // ucp_cleanup(client->ucp_context);

    return ret;
}