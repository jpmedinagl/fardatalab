#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */

#include "ucx_server.h"

#define IP_STRING_LEN 16
#define PORT_STRING_LEN 8

#define CLIENT_SERVER_SEND_RECV_AM UCS_BIT(2)

static struct {
    volatile int complete;
    int          is_rndv;
    void         *desc;
    void         *recv_buf;
} am_data_desc = {0, 0, NULL, NULL};

char * sockaddr_get_ip_str(const struct sockaddr_storage *sock_addr,
                           char *ip_str, size_t max_size)
{
    struct sockaddr_in addr_in;    
    memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
    inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_size);
    return ip_str;
}

char * sockaddr_get_port_str(const struct sockaddr_storage *sock_addr,
                             char *port_str, size_t max_size)
{
    struct sockaddr_in addr_in;

    memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
    snprintf(port_str, max_size, "%d", ntohs(addr_in.sin_port));
    return port_str;
}

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

/**
 * Runs the UCX event loop to process messages.
 */
int ucx_server_run(ucx_server_t *server, send_recv_type_t send_recv_type)
{   
    ucx_connection_t connection;
    ucp_worker_h ucp_data_worker;
    ucp_ep_h server_ep;
    ucs_status_t status;
    int ret;

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

        ret = client_server_do_work(ucp_data_worker, server_ep, send_recv_type, 1);
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
    send_recv_type_t send_recv_type = CLIENT_SERVER_SEND_RECV_AM;
    int ret;

    ucx_server_t server;

    // parse inputs

    ret = ucx_server_init(&server);
    if (ret != 0) {
        return ret;
    }

    ret = ucx_server_run(&server, send_recv_type);
    ucx_server_cleanup(&server);

    return ret;
}