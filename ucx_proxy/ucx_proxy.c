#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

#define SERVER "fardatalab.org"
#define PORT 443
#define REQUEST "GET /research.html HTTP/1.1\r\nHost: fardatalab.org\r\nConnection: close\r\n\r\n"

// Function to initialize OpenSSL
SSL_CTX *init_openssl() {
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    
    const SSL_METHOD *method = TLS_client_method();
    SSL_CTX *ctx = SSL_CTX_new(method);
    if (!ctx) {
        perror("Unable to create SSL context");
        return NULL;
    }
    return ctx;
}

// Function to create and connect a TCP socket
int create_socket(const char *hostname, int port) {
    struct hostent *server;
    struct sockaddr_in server_addr;

    server = gethostbyname(hostname);
    if (!server) {
        fprintf(stderr, "Error: No such host\n");
        return -1;
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        return -1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        return -1;
    }

    return sock;
}

// Function to create SSL structure and connect
SSL *connect_ssl(SSL_CTX *ctx, int sock) {
    SSL *ssl = SSL_new(ctx);
    SSL_set_fd(ssl, sock);

    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        return NULL;
    }
    return ssl;
}

// Function to send an HTTP GET request over SSL
int send_request(SSL *ssl, const char * request, size_t length) {
    if (SSL_write(ssl, request, length) <= 0) {
        perror("Send failed");
        return -1;
    }
    return 0;
}

// Function to receive and print response
void receive_response(SSL *ssl) {
    char buffer[4096];
    printf("Response:\n");
    while (1) {
        int bytes_received = SSL_read(ssl, buffer, sizeof(buffer) - 1);
        if (bytes_received <= 0) break;
        buffer[bytes_received] = '\0';
        printf("%s", buffer);
    }
}

// Function to clean up
void cleanup(SSL *ssl, int sock, SSL_CTX *ctx) {
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(ctx);
}

int proxy_get_request(const char* server, int port, const char *path, size_t length) 
{
    int ret;

    // Initialize OpenSSL
    SSL_CTX *ctx = init_openssl();
    if (ctx == NULL) {
        return -1;
    }

    // Create and connect socket
    int sock = create_socket(server, port);
    if (sock == -1) {
        return -1;
    }

    // Establish SSL connection
    SSL *ssl = connect_ssl(ctx, sock);
    if (ssl == NULL) {
        return -1;
    }

    // Send request and get response
    ret = send_request(ssl, path, length);
    if (ret != 0) {
        return 0;
    }

    receive_response(ssl);

    // Clean up
    cleanup(ssl, sock, ctx);

    return 0;
}

int main() {
    // Initialize OpenSSL
    // SSL_CTX *ctx = init_openssl();

    // // Create and connect socket
    // int sock = create_socket(SERVER, PORT);

    // // Establish SSL connection
    // SSL *ssl = connect_ssl(ctx, sock);

    // // Send request and get response
    // send_request(ssl);
    // receive_response(ssl);

    // // Clean up
    // cleanup(ssl, sock, ctx);
    char server[] = "fardatalab.org";
    int port = 443;
    char request[] = "GET /research.html HTTP/1.1\r\nHost: fardatalab.org\r\nConnection: close\r\n\r\n";
    size_t length = strlen(request);

    proxy_get_request(server, port, request, length);

    return 0;
}
