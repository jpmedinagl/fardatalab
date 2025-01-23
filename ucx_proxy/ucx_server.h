#ifndef UCX_SERVER_H
#define UCX_SERVER_H

#include <ucp/api/ucp.h>

/**
 * ucp context (communication instance)
 * ucp worker (handles communication)
 * ucp listener (listens for connections)
 */
typedef struct ucx_server {
    ucp_context_h * ucp_context;
    ucp_worker_h * ucp_worker;
    ucp_listener_h * ucp_listener;
} ucx_server_t;

typedef int send_recv_type_t;

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
int ucx_server_run(ucx_server_t *server, send_recv_type_t send_recv_type);

/**
 * Cleans up UCX resources.
 */
void ucx_server_cleanup(ucx_server_t *server);

#endif // UCX_SERVER_H
