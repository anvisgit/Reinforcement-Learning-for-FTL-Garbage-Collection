"""
train_ppo.py — drives PPO training/inference by acting as the GC-victim
oracle for the C FTL simulator over the shared-memory IPC bridge.

Usage:
    # Terminal 1: start the C simulator with --rl (it will block waiting
    # for the agent to connect)
    ./ftl_bench --rl

    # Terminal 2: run this script
    python3 scripts/train_ppo.py --episodes 20 --save model.pt

For training we run many short episodes against the C simulator,
restarting the C process between episodes (simplest robust approach
given the C side doesn't reset state mid-process). A wrapper script
(run_training.sh) handles process orchestration.
"""

import argparse
import sys
import time
import torch

sys.path.insert(0, "scripts")
from ipc_client import connect, CMD_PICK, CMD_SHUTDOWN
from ppo_agent import PPOPolicy, PPOTrainer, RolloutBuffer, featurize


def run_episode(policy, frame, max_steps=100000, collect=True):
    buf = RolloutBuffer() if collect else None
    steps = 0
    episode_reward = 0.0

    while steps < max_steps:
        frame.wait_for_command()
        data = frame.read()

        if data["command"] == CMD_SHUTDOWN:
            break
        if data["command"] != CMD_PICK:
            frame.signal_done()
            continue

        g, c, m = featurize(data)
        g_b, c_b, m_b = g.unsqueeze(0), c.unsqueeze(0), m.unsqueeze(0)

        action, logp, value = policy.act(g_b, c_b, m_b)
        action_idx = int(action.item())
        n_cand = data["n_cand"]
        action_idx = min(action_idx, n_cand - 1) if n_cand > 0 else 0
        victim_block = data["cand"][action_idx] if n_cand > 0 else 0

        frame.write_victim(victim_block)

        reward = data["reward"]
        done = bool(data["done"])
        episode_reward += reward

        if collect:
            buf.add(g, c, m, action.squeeze(0), logp.squeeze(0),
                    value.item(), reward, float(done))

        frame.signal_done()
        steps += 1

        if done:
            break

    return buf, episode_reward, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--train", action="store_true",
                    help="run PPO updates; otherwise pure inference")
    args = ap.parse_args()

    policy = PPOPolicy()
    if args.load:
        policy.load_state_dict(torch.load(args.load))
        print(f"Loaded policy from {args.load}")

    trainer = PPOTrainer(policy) if args.train else None

    print("Connecting to FTL shared memory...")
    frame = connect()
    print("Connected. Waiting for simulator commands...")

    for ep in range(args.episodes):
        t0 = time.time()
        buf, ep_reward, steps = run_episode(policy, frame, collect=args.train)
        dt = time.time() - t0

        msg = f"Episode {ep+1}/{args.episodes}: steps={steps} " \
              f"reward={ep_reward:.4f} time={dt:.2f}s"

        if args.train and buf is not None and len(buf) > 0:
            stats = trainer.update(buf)
            msg += f" policy_loss={stats['policy_loss']:.4f} " \
                   f"value_loss={stats['value_loss']:.4f} " \
                   f"entropy={stats['entropy']:.4f}"
        print(msg)

    if args.save:
        torch.save(policy.state_dict(), args.save)
        print(f"Saved policy to {args.save}")

    frame.close()


if __name__ == "__main__":
    main()
