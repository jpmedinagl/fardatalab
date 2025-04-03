#include "peer.cu"

int main() {
    run_peer(0, "192.168.1.2", 1); // GPU 0, initiator
    return 0;
}