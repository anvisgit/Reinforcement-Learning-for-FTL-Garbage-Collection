#include "ipc_bridge.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static int       g_shm_fd  = -1;
static IPCFrame *g_frame   = NULL;
static sem_t    *g_sem_c2py = SEM_FAILED;
static sem_t    *g_sem_py2c = SEM_FAILED;

int ipc_open_server(void) {
    /* clean up stale objects */
    shm_unlink(IPC_SHM_NAME);
    sem_unlink(IPC_SEM_C2PY);
    sem_unlink(IPC_SEM_PY2C);

    g_shm_fd = shm_open(IPC_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (g_shm_fd < 0) { perror("shm_open"); return -1; }
    if (ftruncate(g_shm_fd, IPC_SHM_SIZE) < 0) { perror("ftruncate"); return -1; }

    g_frame = mmap(NULL, IPC_SHM_SIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED, g_shm_fd, 0);
    if (g_frame == MAP_FAILED) { perror("mmap"); return -1; }
    memset(g_frame, 0, IPC_SHM_SIZE);

    g_sem_c2py = sem_open(IPC_SEM_C2PY, O_CREAT, 0666, 0);
    g_sem_py2c = sem_open(IPC_SEM_PY2C, O_CREAT, 0666, 0);
    if (g_sem_c2py == SEM_FAILED || g_sem_py2c == SEM_FAILED) {
        perror("sem_open"); return -1;
    }
    return 0;
}

void ipc_close_server(void) {
    if (g_frame) {
        g_frame->command = IPC_CMD_SHUTDOWN;
        sem_post(g_sem_c2py);   /* wake Python so it can exit */
        usleep(100000);
        munmap(g_frame, IPC_SHM_SIZE);
    }
    if (g_shm_fd >= 0) { close(g_shm_fd); shm_unlink(IPC_SHM_NAME); }
    if (g_sem_c2py != SEM_FAILED) { sem_close(g_sem_c2py); sem_unlink(IPC_SEM_C2PY); }
    if (g_sem_py2c != SEM_FAILED) { sem_close(g_sem_py2c); sem_unlink(IPC_SEM_PY2C); }
}

IPCFrame *ipc_frame(void) { return g_frame; }

uint32_t ipc_call_agent(float *features, uint32_t n_feat,
                         uint32_t *cand,    uint32_t n_cand,
                         double reward,     uint32_t done) {
    if (!g_frame) return cand[0];

    g_frame->n_feat  = n_feat;
    g_frame->n_cand  = n_cand;
    g_frame->reward  = reward;
    g_frame->done    = done;
    memcpy(g_frame->features, features, n_feat * sizeof(float));
    memcpy(g_frame->cand,     cand,     n_cand * sizeof(uint32_t));
    g_frame->command = IPC_CMD_PICK;

    sem_post(g_sem_c2py);    /* signal Python agent */
    sem_wait(g_sem_py2c);    /* wait for response   */

    return g_frame->victim;
}
