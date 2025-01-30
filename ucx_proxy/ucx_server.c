#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#include "ucx_server.h"
#include "ucx_util.h"

#define IP_STRING_LEN 16
#define PORT_STRING_LEN 8

#define AM_ID 0

#define CLIENT_SERVER_SEND_RECV_AM UCS_BIT(2)
#define send_recv_type CLIENT_SERVER_SEND_RECV_AM;

static int connection_closed = 1;

typedef struct {
    int complete;          /* Indicates whether the message processing is complete */
    size_t path_length;    /* Length of the received path */
    char path[1024];       /* Buffer to store the received path */
} am_data_desc_t;

/* Global instance of the structure */
am_data_desc_t am_data_desc = {0};

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

    // initialize context using ucp
    status = ucp_init(&ucp_params, NULL, &(server->ucp_context));
    if (status != UCS_OK) {
        fprintf(stderr, "failed ucp_init");
        ret = -1;
        return ret;
    }

    // initalize worker 
    ret = init_worker(server->ucp_context, &(server->ucp_worker));
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

void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg)
{
    ucx_connection_t * connection = arg;
    ucp_conn_request_attr_t attr;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];
    ucs_status_t status;

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    status = ucp_conn_request_query(conn_request, &attr);
    if (status == UCS_OK) {
        printf("Server received a connection request from client at address %s:%s\n",
               sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
               sockaddr_get_port_str(&attr.client_address, port_str, sizeof(port_str)));
    } else if (status != UCS_ERR_UNSUPPORTED) {
        fprintf(stderr, "failed to query the connection request\n");
    }

    if (connection->conn_request == NULL) {
        connection->conn_request = conn_request;
    } else {
        /* The server is already handling a connection request from a client,
         * reject this new one */
        printf("Rejecting a connection request. Only one client at a time is supported.\n");
        status = ucp_listener_reject(connection->ucp_listener, conn_request);
        if (status != UCS_OK) {
            fprintf(stderr, "server failed to reject a connection request\n");
        }
    }
}

ucs_status_t start_server(ucx_server_t *server, ucx_connection_t *connection, ucp_listener_h *listener_p)
{
    struct sockaddr_storage listen_addr;
    ucp_listener_params_t params;
    ucp_listener_attr_t attr;
    ucs_status_t status;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];

    set_sock_addr(&listen_addr);

    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen = sizeof(listen_addr);
    params.conn_handler.cb = server_conn_handle_cb;
    params.conn_handler.arg = connection;

    status = ucp_listener_create(server->ucp_worker, &params, listener_p);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to listen\n");
        return status;
    }

    attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    status = ucp_listener_query(*listener_p, &attr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to query the listener\n");
        ucp_listener_destroy(*listener_p);
        return status;
    }

    fprintf(stderr, "server is listening on IP %s port %s\n",
            sockaddr_get_ip_str(&attr.sockaddr, ip_str, IP_STRING_LEN),
            sockaddr_get_port_str(&attr.sockaddr, port_str, PORT_STRING_LEN));

    printf("Waiting for connection...\n");

    return status;
}


// ucs_status_t ucp_am_data_cb(void *arg, const void *header, size_t header_length,
//                             void *data, size_t length,
//                             const ucp_am_recv_param_t *param)
// {
//     if (length == 0) {
//         fprintf(stderr, "Received empty data.\n");
//         return UCS_ERR_INVALID_PARAM;
//     }

//     // Here, we assume `data` is the path sent by the client
//     char *received_path = (char *)data;
//     received_path[length] = '\0';  // Null-terminate the string

//     printf("Received path: %s\n", received_path);

//     // Process the path as needed
//     // For example, you can open a file or send a response based on the path

//     // char *response = fetch_data_from_university(received_path);

//     // // Send the fetched data back to the client
//     // ucp_am_send_data(arg, AM_ID, response, strlen(response));

//     // // Clean up and free memory
//     // free(response);


//     // Prepare the success message and path to send back
//     char response[1024];
//     snprintf(response, sizeof(response), "Success: Path received - %s", received_path);

//     // Send the success message back to the client
//     ucp_worker_h ucp_worker = (ucp_worker_h)arg;  // Cast back the worker

//     // Send the response back using AM
//     ucs_status_t status = ucp_am_send_data_nbx(ucp_worker, param->recv_ep, AM_ID, response, strlen(response) + 1);
//     if (status != UCS_OK) {
//         fprintf(stderr, "Failed to send AM message: %s\n", ucs_status_string(status));
//         return UCS_ERR_IO_ERROR;
//     }

//     return UCS_OK;
// }


void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
    connection_closed = 1;
}

ucs_status_t server_create_ep(ucp_worker_h data_worker,
                                     ucp_conn_request_h conn_request,
                                     ucp_ep_h *server_ep)
{
    ucp_ep_params_t ep_params;
    ucs_status_t    status;
    
    printf("server_create_ep\n");

    /* Server creates an ep to the client on the data worker.
     * This is not the worker the listener was created on.
     * The client side should have initiated the connection, leading
     * to this ep's creation */
    ep_params.field_mask      = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request    = conn_request;
    ep_params.err_handler.cb  = err_cb;
    ep_params.err_handler.arg = NULL;

    status = ucp_ep_create(data_worker, &ep_params, server_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create an endpoint on the server: (%s)\n",
                ucs_status_string(status));
    }

    return status;
}

ucs_status_t ucp_am_data_cb(void *arg, const void *header, size_t header_length,
                            void *data, size_t length,
                            const ucp_am_recv_param_t *param)
{
    printf("ucp_am_data_cb\n");

    /* Validate the header */
    if (header_length != sizeof(size_t)) {
        fprintf(stderr, "Received wrong header length %ld (expected %ld)\n",
                header_length, sizeof(size_t));
        return UCS_OK;
    }

    /* Extract the path length from the header */
    am_data_desc.path_length = *(size_t *)header;

    /* Validate the data length */
    if (length != am_data_desc.path_length) {
        fprintf(stderr, "Received wrong data length %ld (expected %ld)\n",
                length, am_data_desc.path_length);
        return UCS_OK;
    }

    /* Store the received path */
    memcpy(am_data_desc.path, data, am_data_desc.path_length);
    am_data_desc.path[am_data_desc.path_length] = '\0'; // Null-terminate the string
    printf("Received path: %s\n", am_data_desc.path);

    /* Mark the operation as complete */
    am_data_desc.complete++;

    return UCS_OK;
}

ucs_status_t register_am_recv_callback(ucp_worker_h worker)
{
    ucp_am_handler_param_t param;

    printf("register_am_recv_callback\n");

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = AM_ID;
    param.cb         = ucp_am_data_cb;
    param.arg        = worker; // don't need worker when receiving the initial path

    return ucp_worker_set_am_recv_handler(worker, &param);
}

ucs_status_t send_data_back(ucp_worker_h worker, ucp_ep_h ep)
{
    
    printf("send_data_back\n");

    ucs_status_ptr_t status_ptr;
    ucs_status_t status;

    if (am_data_desc.path_length == 0) {
        fprintf(stderr, "No data to send.\n");
        return UCS_ERR_INVALID_PARAM;
    }

    printf("Sending path back to client: %s\n", am_data_desc.path);

    status_ptr = ucp_am_send_nb(
        ep, 
        AM_ID,                            // Active Message ID
        am_data_desc.path,                // Data buffer
        am_data_desc.path_length,         // Data size
        ucp_dt_make_contig(1),            // Contiguous data type
        NULL,                             // No callback function
        0                                 // No special flags
    );

    if (UCS_PTR_IS_ERR(status_ptr)) {
        status = UCS_PTR_STATUS(status_ptr);
        fprintf(stderr, "Failed to send data back: %s\n", ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}


int server_do_work(ucp_worker_h ucp_worker, ucp_ep_h ep)
{
    
    printf("server_do_work\n");

    int ret = 0;
    ucs_status_t status;

    // For the server, register the AM callback for receiving messages (paths)
    status = register_am_recv_callback(ucp_worker);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to register AM callback.\n");
        ret = -1;
        return ret;
    }

    // Main loop to keep the server running and handle client messages
    while (!connection_closed) {
        // Process incoming events (non-blocking)
        ucp_worker_progress(ucp_worker);

        // Check if a message has been received and processed
        // if (am_data_desc.complete) {
        //     // Send the same path back to the client
        //     status = send_data_back(ucp_worker, ep);
        //     if (status != UCS_OK) {
        //         fprintf(stderr, "Failed to send data back to client.\n");
        //         ret = -1;
        //         break;
        //     }

        //     // Reset the completion flag
        //     am_data_desc.complete = 0;
        // }
    }

    // Once connection is closed, handle any cleanup
    printf("Server connection closed.\n");

    return ret;
}

/**
 * Runs the UCX event loop to process messages.
 */
int ucx_server_run(ucx_server_t *server)
{   
    ucx_connection_t connection;
    ucp_worker_h ucp_data_worker;
    ucp_ep_h server_ep;
    ucs_status_t status;
    int ret;
    
    printf("ucx_server_run.\n");

    ret = init_worker(server->ucp_context, &ucp_data_worker);
    if (ret != 0) {
        return ret;
    }

    status = register_am_recv_callback(ucp_data_worker);
    if (status != UCS_OK) {
        // CLEANUP
        return -1;
    }

    connection.conn_request = NULL;

    status = start_server(server, &connection, &connection.ucp_listener);
    if (status != UCS_OK) {
        // CLEANUP
        return -1;
    }

    while (1) {
        while(connection.conn_request == NULL) {
            ucp_worker_progress(server->ucp_worker);
        }

        status = server_create_ep(ucp_data_worker, connection.conn_request, &server_ep);
        if (status != UCS_OK) {
            // CLEANUP
            return -1;
        }

        ret = server_do_work(ucp_data_worker, server_ep);
        if (ret != UCS_OK) {
            // CLEANUP
            return -1;
        }

        ep_close(ucp_data_worker, server_ep, UCP_EP_CLOSE_FLAG_FORCE);

        connection.conn_request = NULL;

        printf("waiting for connection...");
    }

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
        ucp_worker_destroy(server->ucp_worker);
    }
    // cleanup context using ucp, free space
    ucp_cleanup(server->ucp_context);
    free(server);
}

int main(int argc, char **argv)
{   
    // Active Message communication choosen
    int ret;

    ucx_server_t server;

    ret = ucx_server_init(&server);
    if (ret != 0) {
        return ret;
    }

    ret = ucx_server_run(&server);
    ucx_server_cleanup(&server);

    return ret;
}