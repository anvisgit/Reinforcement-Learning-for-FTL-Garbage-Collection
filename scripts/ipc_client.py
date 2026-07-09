"""
ipc_client.py — Python side of the shared-memory bridge to the C FTL.

Mirrors the IPCFrame struct in include/ipc_bridge.h exactly:

    uint32_t command;
    uint32_t n_cand;
    uint32_t cand[64];
    uint32_t victim;
    uint32_t n_feat;
    float    features[200];
    double   reward;
    uint32_t done;

All packed, little-endian, no padding (the C struct uses
__attribute__((packed))).
"""

import mmap
import os
import struct
import posix_ipc

SHM_NAME   = "/ftl_rl_shm"
SEM_C2PY   = "/ftl_rl_c2py"
SEM_PY2C   = "/ftl_rl_py2c"

CMD_IDLE     = 0
CMD_PICK     = 1
CMD_SHUTDOWN = 2

MAX_CAND = 64
MAX_FEAT = 200

# struct format: packed little-endian
#   I command
#   I n_cand
#   64I cand
#   I victim
#   I n_feat
#   200f features
#   d reward
#   I done
FMT = "<II" + f"{MAX_CAND}I" + "II" + f"{MAX_FEAT}f" + "dI"
FRAME_SIZE = struct.calcsize(FMT)


class IPCFrame:
    """Read/write the shared-memory frame using struct packing."""

    def __init__(self):
        self.shm = posix_ipc.SharedMemory(SHM_NAME)
        self.mm  = mmap.mmap(self.shm.fd, FRAME_SIZE)
        os.close(self.shm.fd)
        self.sem_c2py = posix_ipc.Semaphore(SEM_C2PY)
        self.sem_py2c = posix_ipc.Semaphore(SEM_PY2C)

    def read(self):
        self.mm.seek(0)
        raw = self.mm.read(FRAME_SIZE)
        vals = struct.unpack(FMT, raw)
        command  = vals[0]
        n_cand   = vals[1]
        cand     = vals[2:2 + MAX_CAND]
        victim   = vals[2 + MAX_CAND]
        n_feat   = vals[3 + MAX_CAND]
        features = vals[4 + MAX_CAND: 4 + MAX_CAND + MAX_FEAT]
        reward   = vals[4 + MAX_CAND + MAX_FEAT]
        done     = vals[5 + MAX_CAND + MAX_FEAT]
        return {
            "command": command,
            "n_cand": n_cand,
            "cand": list(cand[:n_cand]),
            "victim": victim,
            "n_feat": n_feat,
            "features": list(features[:n_feat]),
            "reward": reward,
            "done": done,
        }

    def write_victim(self, victim: int):
        # only need to overwrite the `victim` field
        offset = struct.calcsize("<II" + f"{MAX_CAND}I")
        self.mm.seek(offset)
        self.mm.write(struct.pack("<I", victim))

    def wait_for_command(self, timeout=None):
        self.sem_c2py.acquire(timeout)

    def signal_done(self):
        self.sem_py2c.release()

    def close(self):
        self.mm.close()


def connect(retries=50, delay=0.1):
    """Connect to the shared memory segment, retrying until C side opens it."""
    import time
    last_err = None
    for _ in range(retries):
        try:
            return IPCFrame()
        except posix_ipc.ExistentialError as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to FTL shared memory: {last_err}")
