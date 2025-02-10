#ifndef UCP_UTIL_H_
#define UCP_UTIL_H_

#include <ucp/api/ucp.h>

#include <string.h>    /* memset */
#include <arpa/inet.h> /* inet_addr */
#include <unistd.h>    /* getopt */
#include <stdlib.h>    /* atoi */


/**
 * Closes a UCX endpoint with the specified flags.  
 * Uses non-blocking close and waits for completion before freeing the request.  
 */
void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep, uint64_t flags);

/**
 * Extracts and returns the IP address as a string from a sockaddr_storage structure.
 */
char * sockaddr_get_ip_str(const struct sockaddr_storage *sock_addr,
                           char *ip_str, size_t max_size);

/**
 * Extracts and returns the port number as a string from a sockaddr_storage structure.
 */
char * sockaddr_get_port_str(const struct sockaddr_storage *sock_addr,
                             char *port_str, size_t max_size);

#endif