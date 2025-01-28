#ifndef UCX_SERVER_H
#define UCX_SERVER_H

#include <ucp/api/ucp.h>

/**
 * ucp context (communication instance)
 * ucp worker (handles communication)
 */
typedef struct ucx_server {
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
} ucx_server_t;

typedef struct ucx_connection {
    volatile ucp_conn_request_h conn_request;
    ucp_listener_h ucp_listener;
} ucx_connection_t;

/**
 * Initializes the UCX server by setting up context and worker.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_init(ucx_server_t *server);

/**
 * Starts the UCX listener to accept incoming client connections.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_server_listen(ucx_server_t *server, const char *ip, int port);

/**
 * Runs the UCX event loop to process messages.
 */
int ucx_server_run(ucx_server_t *server);

/**
 * Cleans up UCX resources.
 */
void ucx_server_cleanup(ucx_server_t *server);

#endif // UCX_SERVER_H
