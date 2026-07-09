"""
train_loop.py — orchestrates multi-episode PPO training.

For each episode:
  1. Spawn `./ftl_bench --rl` (fresh NAND device, blocks on IPC)
  2. Connect Python agent, run episode, collect rollout, PPO update
  3. Wait for C process to exit, repeat

Much shorter workload per episode than the final benchmark, so training
is fast; final evaluation uses the full workload via run_eval.sh.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from ipc_client import connect
from ppo_agent import PPOPolicy, PPOTrainer, RolloutBuffer
from train_ppo import run_episode

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN  = os.path.join(ROOT, "ftl_bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-writes", type=int, default=6000,
                    help="host writes per training episode (kept small for speed)")
    ap.add_argument("--save", type=str, default="model.pt")
    ap.add_argument("--log", type=str, default="results/training_log.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_f = open(args.log, "w")
    log_f.write("episode,reward,steps,final_waf,policy_loss,value_loss,entropy,time_s\n")

    policy  = PPOPolicy()
    trainer = PPOTrainer(policy)

    env = os.environ.copy()
    env["FTL_N_WRITES"] = str(args.n_writes)
    env["FTL_SKIP_BASELINES"] = "1"

    for ep in range(args.episodes):
        t0 = time.time()
        proc = subprocess.Popen([BIN, "--rl"], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        time.sleep(0.3)  # let C side open shared memory

        try:
            frame = connect(retries=50, delay=0.05)
        except RuntimeError as e:
            print(f"Episode {ep}: connection failed: {e}")
            proc.kill()
            continue

        buf, ep_reward, steps = run_episode(policy, frame, collect=True)
        frame.close()

        stdout, _ = proc.communicate(timeout=10)
        dt = time.time() - t0

        # extract final WAF from C stdout
        final_waf = None
        for line in stdout.splitlines():
            if "ppo" in line and "WAF=" in line:
                try:
                    final_waf = float(line.split("WAF=")[1].split()[0])
                except (IndexError, ValueError):
                    pass

        if buf is not None and len(buf) > 1:
            n_cands_seen = [int(m.sum().item()) for m in buf.masks]
            avg_cand = sum(n_cands_seen) / len(n_cands_seen)
            stats = trainer.update(buf)
            stats["avg_cand"] = avg_cand
        else:
            stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "avg_cand": 0}

        waf_str = f"{final_waf:.4f}" if final_waf is not None else "NA"
        print(f"Ep {ep+1:3d}/{args.episodes}  steps={steps:5d}  "
              f"reward={ep_reward:8.4f}  WAF={waf_str}  "
              f"p_loss={stats['policy_loss']:.4f}  "
              f"v_loss={stats['value_loss']:.4f}  "
              f"ent={stats['entropy']:.4f}  avg_cand={stats.get('avg_cand',0):.1f}  ({dt:.1f}s)")

        log_f.write(f"{ep+1},{ep_reward},{steps},{final_waf if final_waf else ''},"
                    f"{stats['policy_loss']},{stats['value_loss']},"
                    f"{stats['entropy']},{dt}\n")
        log_f.flush()

        if (ep + 1) % 5 == 0 or ep == args.episodes - 1:
            torch.save(policy.state_dict(), args.save)
            print(f"  [checkpoint saved to {args.save}]")

    log_f.close()
    torch.save(policy.state_dict(), args.save)
    print(f"\nTraining complete. Final model saved to {args.save}")


if __name__ == "__main__":
    main()
