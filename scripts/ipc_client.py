

import socket
import struct
import time

HOST = "127.0.0.1"
PORT = 5555

CMD_IDLE = 0
CMD_PICK = 1
CMD_SHUTDOWN = 2

MAX_CAND = 64
MAX_FEAT = 200

FMT = "<II" + f"{MAX_CAND}I" + "II" + f"{MAX_FEAT}f" + "dI"
FRAME_SIZE = struct.calcsize(FMT)


class IPCFrame:

    def __init__(self):
        self.sock = socket.create_connection((HOST, PORT), timeout=5.0)
        self.sock.settimeout(None)
        self._pending = None

    def _recv_exact(self, n):
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self.sock.recv(n - len(chunks))
            if not chunk:
                raise ConnectionError("socket closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def read(self):
        if self._pending is None:
            raise RuntimeError("No pending frame")
        raw = self._pending
        self._pending = None
        vals = struct.unpack(FMT, raw)
        command = vals[0]
        n_cand = vals[1]
        cand = vals[2:2 + MAX_CAND]
        victim = vals[2 + MAX_CAND]
        n_feat = vals[3 + MAX_CAND]
        features = vals[4 + MAX_CAND: 4 + MAX_CAND + MAX_FEAT]
        reward = vals[4 + MAX_CAND + MAX_FEAT]
        done = vals[5 + MAX_CAND + MAX_FEAT]
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
        offset = struct.calcsize("<II" + f"{MAX_CAND}I")
        frame = bytearray(self._pending or b"\x00" * FRAME_SIZE)
        frame[offset:offset + 4] = struct.pack("<I", victim)
        self.sock.sendall(bytes(frame))

    def wait_for_command(self, timeout=None):
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            self._pending = self._recv_exact(FRAME_SIZE)
        finally:
            self.sock.settimeout(None)

    def signal_done(self):
        return None

    def close(self):
        self.sock.close()


def connect(retries=50, delay=0.1):
    """Connect to the FTL benchmark, retrying until the C side opens the socket."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return IPCFrame()
        except OSError as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay)
                continue
    raise RuntimeError(f"Could not connect to FTL benchmark socket on {HOST}:{PORT}: {last_err}")
