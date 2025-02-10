#include "ucx_util.h"

/**
 * Closes a UCX endpoint with the specified flags.  
 * Uses non-blocking close and waits for completion before freeing the request.  
 */
void ep_close(ucp_worker_h ucp_worker, ucp_ep_h ep, uint64_t flags) 
{
    ucp_request_param_t param;
    ucs_status_t status;
    void *close_req;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags = flags;
    close_req = ucp_ep_close_nbx(ep, &param);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(close_req);
    } else {
        status = UCS_PTR_STATUS(close_req);
    }

    if (status != UCS_OK) {
        fprintf(stderr, "failed to close ep \n");
    }
}

/**
 * Extracts and returns the IP address as a string from a sockaddr_storage structure.
 */
char * sockaddr_get_ip_str(const struct sockaddr_storage *sock_addr,
                           char *ip_str, size_t max_size)
{
    struct sockaddr_in addr_in;
    memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
    inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_size);
    return ip_str;
}

/**
 * Extracts and returns the port number as a string from a sockaddr_storage structure.
 */
char * sockaddr_get_port_str(const struct sockaddr_storage *sock_addr,
                             char *port_str, size_t max_size)
{
    struct sockaddr_in addr_in;
    memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
    snprintf(port_str, max_size, "%d", ntohs(addr_in.sin_port));
    return port_str;
}