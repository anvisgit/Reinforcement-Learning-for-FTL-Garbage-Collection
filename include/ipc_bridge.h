#ifndef IPC_BRIDGE_H
#define IPC_BRIDGE_H

/*
 * Shared-memory IPC between the C FTL simulator and the Python PPO agent.
 *
 * Layout of the shared segment (all little-endian):
 *   [0]      : uint32  command  (0=idle, 1=pick_victim, 2=shutdown)
 *   [4]      : uint32  n_cand   (number of candidate block ids)
 *   [8]      : uint32  cand[64] (candidate block ids)
 *   [8+256]  : uint32  victim   (result written by agent)
 *   [8+260]  : uint32  n_feat
 *   [8+264]  : float   features[200]
 *   [8+1064] : double  reward   (previous step reward, written by C)
 *   [8+1072] : uint32  done     (episode done flag)
 */

#include <stdint.h>
#include <stddef.h>

#define IPC_SHM_NAME   "/ftl_rl_shm"
#define IPC_SEM_C2PY   "/ftl_rl_c2py"   /* C posts, Python waits  */
#define IPC_SEM_PY2C   "/ftl_rl_py2c"   /* Python posts, C waits  */

#define IPC_CMD_IDLE     0
#define IPC_CMD_PICK     1
#define IPC_CMD_SHUTDOWN 2

#define IPC_MAX_CAND   64
#define IPC_MAX_FEAT  200

typedef struct __attribute__((packed)) {
    uint32_t command;
    uint32_t n_cand;
    uint32_t cand[IPC_MAX_CAND];
    uint32_t victim;
    uint32_t n_feat;
    float    features[IPC_MAX_FEAT];
    double   reward;
    uint32_t done;
} IPCFrame;

#define IPC_SHM_SIZE sizeof(IPCFrame)

#ifdef __cplusplus
extern "C" {
#endif

/* C-side: open / close shared memory + semaphores */
int  ipc_open_server(void);   /* returns fd or -1 */
void ipc_close_server(void);
IPCFrame *ipc_frame(void);    /* pointer to shared frame */

/* Block until Python agent posts victim; returns victim block id */
uint32_t ipc_call_agent(float *features, uint32_t n_feat,
                         uint32_t *cand, uint32_t n_cand,
                         double reward, uint32_t done);

#ifdef __cplusplus
}
#endif

#endif /* IPC_BRIDGE_H */
