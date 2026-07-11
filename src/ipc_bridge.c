#include "ipc_bridge.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#define IPC_HOST "127.0.0.1"
#define IPC_PORT 5555

static int g_listen_fd = -1;
static int g_client_fd = -1;
static IPCFrame g_frame;

static int socket_close(int fd) {
#if defined(_WIN32)
    return closesocket(fd);
#else
    return close(fd);
#endif
}

static int socket_init(void) {
#if defined(_WIN32)
    static int ws_started = 0;
    if (!ws_started) {
        WSADATA wsa;
        int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (rc != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", rc);
            return -1;
        }
        ws_started = 1;
    }
#endif
    return 0;
}

int ipc_open_server(void) {
    if (socket_init() < 0) return -1;
    if (g_listen_fd >= 0) return 0;

    g_listen_fd = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd < 0) { perror("socket"); return -1; }

    int opt = 1;
    setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(IPC_PORT);

    if (bind(g_listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        socket_close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    if (listen(g_listen_fd, 1) < 0) {
        perror("listen");
        socket_close(g_listen_fd);
        g_listen_fd = -1;
        return -1;
    }

    memset(&g_frame, 0, sizeof(g_frame));
    return 0;
}

void ipc_close_server(void) {
    if (g_client_fd >= 0) {
        socket_close(g_client_fd);
        g_client_fd = -1;
    }
    if (g_listen_fd >= 0) {
        socket_close(g_listen_fd);
        g_listen_fd = -1;
    }
}

IPCFrame *ipc_frame(void) { return &g_frame; }

uint32_t ipc_call_agent(float *features, uint32_t n_feat,
                         uint32_t *cand,    uint32_t n_cand,
                         double reward,     uint32_t done) {
    if (!cand || n_cand == 0) return 0;

    if (g_client_fd < 0) {
        g_client_fd = accept(g_listen_fd, NULL, NULL);
        if (g_client_fd < 0) {
            perror("accept");
            return cand[0];
        }
    }

    memset(&g_frame, 0, sizeof(g_frame));
    g_frame.command = IPC_CMD_PICK;
    g_frame.n_feat  = n_feat;
    g_frame.n_cand  = n_cand;
    g_frame.reward  = reward;
    g_frame.done    = done;
    if (features && n_feat > 0) {
        memcpy(g_frame.features, features, n_feat * sizeof(float));
    }
    if (cand && n_cand > 0) {
        memcpy(g_frame.cand, cand, n_cand * sizeof(uint32_t));
    }

    if (send(g_client_fd, (const char *)&g_frame, sizeof(g_frame), 0) != (int)sizeof(g_frame)) {
        perror("send");
        return cand[0];
    }

    int received = recv(g_client_fd, (char *)&g_frame, sizeof(g_frame), 0);
    if (received != (int)sizeof(g_frame)) {
        perror("recv");
        return cand[0];
    }

    return g_frame.victim;
}
