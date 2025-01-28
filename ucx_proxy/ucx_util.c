#include "ucx_util.h"

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