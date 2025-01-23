#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#include "ucx_server.h"

#define CLIENT_SERVER_SEND_RECV_AM UCS_BIT(2)

/**
 * Handles a request to the server. Callback function. In this simple 
 * client-server model, the server does not do anything except send ok. Later
 * when doing proxy, server will send a request to the fardatalab website.
 */
// ucs_status_t handle_request(void *arg, const void *data, size_t length, 
//                             ucp_ep_h reply_ep, unsigned flags) {
//     // Incoming request
//     printf("Request: %.*s\n", (int)length, (const char *)data);

//     const char *response = "Response:";
    
//     // Send response
//     ucp_request_param_t param = {0};
//     ucs_status_ptr_t send_status = ucp_am_send_nb(reply_ep, 0, response, 
//                                                   strlen(response) + 1, 
//                                                   UCP_AM_SEND_FLAG_EAGER, NULL);
//     if (UCS_PTR_IS_ERR(send_status)) {
//         fprintf(stderr, "Error sending response\n");
//         return UCS_ERR_NO_MEMORY;
//     }
//     return UCS_OK;
// }

/**
 * Initialize the worker for this server. We are using a single thread to 
 * process incoming requests.
 */
int init_worker(ucx_server_t *server) 
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;
    int ret = 0;

    memset(&worker_params, 0, sizeof(worker_params));

    // worker uses single thread mode
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    // allocate worker space
    server->ucp_worker = malloc(sizeof(ucp_worker_h));
    if (server->ucp_worker == NULL) {
        fprintf(stderr, "malloc failed\n");
        ret = -1;
        return ret;
    }

    // create worker using ucp
    status = ucp_worker_create(*(server->ucp_context), &worker_params, server->ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "failed ucp_worker_create\n");
        ret = -1;
    }
    return ret;
}

/**
 * Initializes the UCX server by setting up context and worker.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_init(ucx_server_t *server) 
{
    ucp_params_t ucp_params;
    ucs_status_t status;
    int ret = 0;

    memset(&ucp_params, 0, sizeof(ucp_params));

    // feature types: TAG, STREAM, RMA, AM -> AM for this server
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_AM;

    // allocate context space
    server->ucp_context = malloc(sizeof(ucp_context_h));
    if (server->ucp_context == NULL) {
        fprintf(stderr, "malloc failed\n");
        ret = -1;
        return ret;
    }

    // initialize context using ucp
    status = ucp_init(&ucp_params, NULL, server->ucp_context);
    if (status != UCS_OK) {
        fprintf(stderr, "failed ucp_init");
        ret = -1;
        return ret;
    }

    // initalize worker 
    ret = init_worker(server);
    if (ret != 0) {
        ucx_server_cleanup(server);
    }
    
    return ret;
}

/**
 * Starts the UCX listener to accept incoming client connections.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_listen(ucx_server_t *server, const char *ip, int port)
{
    return 0;
}

/**
 * Runs the UCX event loop to process messages.
 */
int ucx_server_run(ucx_server_t *server, send_recv_type_t send_recv_type)
{
    return 0;
}

/**
 * Cleans up UCX resources. Cleans up both the ucp context as well as the ucp 
 * worker servicing the context.
 */
void ucx_server_cleanup(ucx_server_t *server) 
{
    // worker may not have been initialized (incorrect context initialization)
    if (server->ucp_worker != NULL) {
        ucp_worker_destroy(*(server->ucp_worker));
        free(server->ucp_worker);
    }
    // cleanup context using ucp, free space
    ucp_cleanup(*(server->ucp_context));
    free(server->ucp_context);
}

int main(int argc, char **argv) 
{   
    // Active Message communication choosen    
    send_recv_type_t send_recv_type = CLIENT_SERVER_SEND_RECV_AM;
    // char *listen_addr = NULL;
    int ret;

    ucx_server_t server;
    server.ucp_context = NULL;
    server.ucp_worker = NULL;
    server.ucp_listener = NULL;

    // parse inputs

    ret = ucx_server_init(&server);
    if (ret != 0) {
        return ret;
    }

    ret = ucx_server_run(&server, send_recv_type);
    ucx_server_cleanup(&server);

    return ret;
}