#ifndef UCX_CLIENT_H
#define UCX_CLIENT_H

#include <ucp/api/ucp.h>

/**
 * UCX client structure containing context and worker.
 */
typedef struct ucx_client {
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_ep_h ucp_ep;
} ucx_client_t;

/**
 * Initializes the UCX client by setting up context and worker.
 * Returns 0 on success, non-zero on failure.
 */
int ucx_client_init(ucx_client_t *client);

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
int ucx_client_receive(ucx_client_t *client);

/**
 * Cleans up UCX resources.
 */
void ucx_client_cleanup(ucx_client_t *client);

#endif // UCX_CLIENT_H
