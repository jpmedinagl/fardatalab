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

/**
 * Initialize the worker for this server. We are using a single thread to 
 * process incoming requests.
 */
int worker_init(ucp_context_h ucp_context, ucp_worker_h *ucp_worker)
{
    ucp_worker_params_t worker_params;
    ucs_status_t status;

    memset(&worker_params, 0, sizeof(worker_params));

    // worker uses single thread mode
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    // create worker using ucp
    status = ucp_worker_create(ucp_context, &worker_params, ucp_worker);
    if (UCS_STATUS_IS_ERR(status)) {
        fprintf(stderr, "failed ucp_worker_create: %s\n", ucs_status_string(status));
        return -1;
    }
    return 0;
}

/**
 * Initializes the UCX server by setting up context and worker.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_init(ucx_server_t *server) 
{
    ucp_params_t ucp_params;
    ucs_status_t status;

    memset(&ucp_params, 0, sizeof(ucp_params));

    // feature types: TAG, STREAM, RMA, AM -> AM for this server
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_AM;

    // initialize context using ucp
    status = ucp_init(&ucp_params, NULL, &(server->ucp_context));
    if (UCS_STATUS_IS_ERR(status)) {
        fprintf(stderr, "failed ucp_init: %s\n", ucs_status_string(status));
        return -1;
    }

    // initalize worker 
    int ret = worker_init(server->ucp_context, &(server->ucp_worker));
    if (ret != 0) {
        ucx_server_cleanup(server);
        return -1;
    }
    
    return 0;
}

/**
 * Callback for handling incoming connection requests from clients.  
 * If the server is not already handling a connection, it accepts the new request.  
 * If a connection is already being handled, it rejects the new request.  
 */
void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg)
{
    ucx_connection_t * connection = arg;
    ucp_conn_request_attr_t attr;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];
    ucs_status_t status;

    memset(&attr, 0, sizeof(attr));
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    status = ucp_conn_request_query(conn_request, &attr);
    if (status == UCS_OK) {
        printf("Server received a connection request from client at address %s:%s\n",
               sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
               sockaddr_get_port_str(&attr.client_address, port_str, sizeof(port_str)));
        
    } else if (status != UCS_ERR_UNSUPPORTED) {
        fprintf(stderr, "failed to query the connection request\n");
        return ;
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

/**
 * Initializes a sockaddr_storage structure with default server address (INADDR_ANY)  
 * and port.  
 */
void set_sock_addr(struct sockaddr_storage *saddr)
{
    struct sockaddr_in * sa_in;

    memset(saddr, 0, sizeof(*saddr));

    sa_in = (struct sockaddr_in*)saddr;
    sa_in->sin_addr.s_addr = INADDR_ANY;
    sa_in->sin_family = AF_INET;
    sa_in->sin_port = htons(DEFAULT_PORT);
}

/**
 * Sets up socket. Starts the UCX listener to accept incoming client connections.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_listen(ucx_server_t *server, ucx_connection_t *connection)
{
    struct sockaddr_storage listen_addr;
    ucp_listener_params_t params;
    ucp_listener_attr_t attr;
    ucs_status_t status;
    char ip_str[IP_STRING_LEN];
    char port_str[PORT_STRING_LEN];

    set_sock_addr(&listen_addr);

    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen = sizeof(listen_addr);
    params.conn_handler.cb = server_conn_handle_cb;
    params.conn_handler.arg = connection;

    status = ucp_listener_create(server->ucp_worker, &params, &(connection->ucp_listener));
    if (UCS_STATUS_IS_ERR(status)) {
        fprintf(stderr, "failed to listen: %s\n", ucs_status_string(status));
        return status;
    }

    memset(&attr, 0, sizeof(attr));
    attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    status = ucp_listener_query(connection->ucp_listener, &attr);
    if (UCS_STATUS_IS_ERR(status)) {
        fprintf(stderr, "failed to query the listener: %s\n", ucs_status_string(status));
        ucp_listener_destroy(connection->ucp_listener);
        return -1;
    }

    fprintf(stderr, "server is listening on IP %s port %s\n",
            sockaddr_get_ip_str(&attr.sockaddr, ip_str, IP_STRING_LEN),
            sockaddr_get_port_str(&attr.sockaddr, port_str, PORT_STRING_LEN));

    printf("\n\nWaiting for connection...\n");
    printf("----------------------------------------------------\n");

    return 0;
}

/**
 * Callback function for error handling. Used when server could not satisfy
 * connection request.
 */
void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    printf("error handling callback was invoked with status %d (%s)\n",
           status, ucs_status_string(status));
    connection_closed = 1;
}

/**
 * Creates a endpoint (ep) for communication on the server on the data worker.  
 * The endpoint is created using a connection request, and an error handler is assigned.  
 * Returns 0 on success, or -1 on error.  
 */
int server_create_ep(ucp_worker_h data_worker,
                     ucp_conn_request_h conn_request,
                     ucp_ep_h *server_ep)
{
    ucp_ep_params_t ep_params;
    ucs_status_t    status;

    memset(&ep_params, 0, sizeof(ep_params));

    /* Server creates an ep to the client on the data worker.
     * This is not the worker the listener was created on.
     * The client side should have initiated the connection, leading
     * to this ep's creation */
    ep_params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLER |
                           UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_request;
    ep_params.err_handler.cb = err_cb;
    ep_params.err_handler.arg = NULL;

    status = ucp_ep_create(data_worker, &ep_params, server_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to create an endpoint on the server: (%s)\n",
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
    connection_data_t *conn_data = (connection_data_t *)arg;

    if (header_length != sizeof(size_t)) {
        fprintf(stderr, "Received wrong header length %ld (expected %ld)\n",
                header_length, sizeof(size_t));
        return UCS_OK;
    }

    // Extract the path length from the header (sent from client)
    conn_data->am_data.path_length = *(size_t *)header;

    if (length != conn_data->am_data.path_length) {
        fprintf(stderr, "Received wrong data length %ld (expected %ld)\n",
                length, conn_data->am_data.path_length);
        return UCS_OK;
    }

    // Store the received path (proxy will process it)
    memcpy(conn_data->am_data.path, data, conn_data->am_data.path_length);
    conn_data->am_data.path[conn_data->am_data.path_length] = '\0'; // Null-terminate the string
    printf("\nReceived data: %s (%ld)\n", conn_data->am_data.path, conn_data->am_data.path_length);

    // Operation as complete
    conn_data->am_data.complete = 1;

    return UCS_OK;
}

/**
 * Registers a callback to handle incoming active messages on the worker.  
 * Associates the message ID with the callback function for processing received data.  
 */
ucs_status_t register_am_recv_callback(ucp_worker_h worker, connection_data_t *conn_data)
{
    ucp_am_handler_param_t param;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id = AM_ID;
    param.cb = recv_am_data_cb;
    param.arg = conn_data;

    return ucp_worker_set_am_recv_handler(worker, &param);
}

/**
 * Callback function for handling the completion of an active messages send operation.  
 * Frees the request and logs an error if the send operation fails.  
 */
void send_callback(void *request, ucs_status_t status, void *user_data)
 {
    if (status != UCS_OK) {
        fprintf(stderr, "\nSend operation failed: %s\n", ucs_status_string(status));
    } 
    // else {
    //     printf("\nSend operation completed successfully.\n");
    // }

    // Free the request after completion
    ucp_request_free(request);
}

/**
 * Sends an active message to the specified endpoint with a length-prefixed header.  
 * Uses send_callback to handle completion and returns the operation status.  
 */
ucs_status_t send_am_data(ucp_worker_h ucp_worker, ucp_ep_h ep, connection_data_t *conn_data)
{   
    char * msg = conn_data->am_data.path;
    size_t msg_length = conn_data->am_data.path_length;

    if (!ep) {
        fprintf(stderr, "send_am_message: Attempted to use a NULL endpoint!\n");
        return UCS_ERR_INVALID_PARAM;
    }

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
    request = ucp_am_send_nbx(ep, AM_ID, &header, sizeof(size_t), msg, msg_length, &param);

    if (request == NULL) {
        // Operation completed immediately
        return UCS_OK;
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
 * Registers callback for the server. Deals with status.
 */
int register_am_callback_for_server(ucp_worker_h ucp_worker, connection_data_t *conn_data) 
{
    ucs_status_t status = register_am_recv_callback(ucp_worker, conn_data);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to register AM callback.\n");
        return -1;
    }
    return 0;
}

/**
 * Handles incoming messages. Loops until the connection is closed, or client
 * has completed sending data.
 */
int handle_incoming_messages(ucp_worker_h ucp_worker, connection_data_t *conn_data) 
{
    while (!connection_closed && !conn_data->am_data.complete) {
        ucp_worker_progress(ucp_worker);
    }

    return conn_data->am_data.complete ? 0 : -1;  // Return success or failure based on completion
}

/**
 * Sends response to client through the send_am_data function.
 */
int send_response_to_client(ucp_worker_h ucp_worker, ucp_ep_h ep, connection_data_t *conn_data) 
{
    printf("Sending data: %s (%ld)\n", conn_data->am_data.path, conn_data->am_data.path_length);

    ucs_status_t status = send_am_data(ucp_worker, ep, conn_data);
    if (status != UCS_OK) {
        fprintf(stderr, "Failed to send data back to client.\n");
        return -1;
    }
    conn_data->am_data.complete = 0;  // Reset for future messages
    return 0;
}

/**
 * Registers the active messages call back for receiving paths from client.
 * Handles incoming messages. As well as responds to clients. Finishes when
 * connection is closed.
 */
int server_do_work(ucp_worker_h ucp_worker, ucp_ep_h ep, connection_data_t *conn_data)
{
    int ret = 0;

    // Register the AM callback for receiving messages (paths)
    ret = register_am_callback_for_server(ucp_worker, conn_data);
    if (ret != 0) return ret;

    connection_closed = 0;

    // Handle incoming messages
    ret = handle_incoming_messages(ucp_worker, conn_data);
    if (ret != 0) return ret;

    // Send response to client if message is complete
    if (conn_data->am_data.complete) {
        ret = send_response_to_client(ucp_worker, ep, conn_data);
    }

    // Wait for all operations to complete
    while (ucp_worker_progress(ucp_worker)) {
        // Process events until all operations are completed
    }

    connection_closed = 1;
    printf("\nClient connection closed.\n");

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
    connection_data_t conn_data;
    int ret;
    
    memset(&connection, 0, sizeof(ucx_connection_t));

    ret = worker_init(server->ucp_context, &ucp_data_worker);
    if (ret != 0) {
        return ret;
    }

    ret = ucx_server_listen(server, &connection);
    if (ret != 0) {
        ucp_worker_destroy(ucp_data_worker);
        return -1;
    }

    memset(&server_ep, 0, sizeof(ucp_ep_h));
    memset(&conn_data, 0, sizeof(connection_data_t));

    while (1) {
        while (connection.conn_request == NULL) {
            ucp_worker_progress(server->ucp_worker);
        }

        ret = server_create_ep(ucp_data_worker, connection.conn_request, &server_ep);
        if (ret != 0) {
            // Reset connection request
            memset(&conn_data, 0, sizeof(connection_data_t));
            connection.conn_request = NULL;
            continue; // Allow server to keep running
        }

        conn_data.ep = server_ep;

        ret = server_do_work(ucp_data_worker, server_ep, &conn_data);

        // Close end-point clean-up connection data
        ep_close(ucp_data_worker, server_ep, UCP_EP_CLOSE_FLAG_FORCE);

        // Reset connection data for new connection.
        memset(&conn_data, 0, sizeof(connection_data_t));

        // Reset the connection request
        connection.conn_request = NULL;

        printf("\n\nWaiting for connection...\n");
        printf("----------------------------------------------------\n");
        
    }
    
    ucp_worker_destroy(ucp_data_worker);

    return 0;
}

/**
 * Cleans up ucx resources. Cleans up both the ucp context as well as the ucp 
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
}

int main(int argc, char **argv)
{   
    ucx_server_t server;
    int ret;

    ret = ucx_server_init(&server);
    if (ret != 0) {
        // failed to initialize the server correctly -> clean up is handled in the init
        return ret;
    }

    // run server
    ret = ucx_server_run(&server);

    // cleanup server - technically we have an infinite while loop, but good practice
    ucx_server_cleanup(&server);

    return ret;
}