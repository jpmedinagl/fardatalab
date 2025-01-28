#ifndef UCP_UTIL_H_
#define UCP_UTIL_H_

#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */


/**
 * Close UCP endpoint.
 *
 * @param [in]  worker  Handle to the worker that the endpoint is associated
 *                      with.
 * @param [in]  ep      Handle to the endpoint to close.
 * @param [in]  flags   Close UCP endpoint mode. Please see
 *                      @a ucp_ep_close_flags_t for details.
 */
void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep, uint64_t flags);

char * sockaddr_get_ip_str(const struct sockaddr_storage *sock_addr,
                           char *ip_str, size_t max_size);

char * sockaddr_get_port_str(const struct sockaddr_storage *sock_addr,
                             char *port_str, size_t max_size);

void set_sock_addr(struct sockaddr_storage *saddr);

#endif