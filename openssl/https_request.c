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

int main() {
    int sock;
    struct sockaddr_in server_addr;
    struct hostent *server;
    char buffer[4096];

    // Step 1: Initialize OpenSSL
    SSL_library_init();
    SSL_load_error_strings();
    const SSL_METHOD *method = TLS_client_method();
    SSL_CTX *ctx = SSL_CTX_new(method);
    if (!ctx) {
        perror("Unable to create SSL context");
        exit(EXIT_FAILURE);
    }

    // Step 2: Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation failed");
        exit(1);
    }

    // Step 3: Resolve hostname to IP address
    server = gethostbyname(SERVER);
    if (!server) {
        fprintf(stderr, "Error: No such host\n");
        exit(1);
    }

    // Step 4: Set up server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);

    // Step 5: Connect to server
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Connection failed");
        exit(1);
    }

    // Step 6: Create SSL structure and attach to socket
    SSL *ssl = SSL_new(ctx);
    SSL_set_fd(ssl, sock);

    // Step 7: Perform SSL handshake
    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        exit(1);
    }

    // Step 8: Send HTTP GET request over SSL
    if (SSL_write(ssl, REQUEST, strlen(REQUEST)) <= 0) {
        perror("Send failed");
        exit(1);
    }

    // Step 9: Receive and print response
    printf("Response:\n");
    while (1) {
        int bytes_received = SSL_read(ssl, buffer, sizeof(buffer) - 1);
        if (bytes_received <= 0) {
            break; // Connection closed or error
        }
        buffer[bytes_received] = '\0';
        printf("%s", buffer);
    }

    // Step 10: Cleanup
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(ctx);

    return 0;
}

