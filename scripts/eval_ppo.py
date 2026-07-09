"""
eval_ppo.py — runs the trained PPO policy in pure inference mode
against the C simulator for the final benchmark comparison.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ipc_client import connect
from ppo_agent import PPOPolicy
from train_ppo import run_episode


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "results/ppo_model.pt"
    policy = PPOPolicy()
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    print(f"Loaded trained policy from {model_path}")

    frame = connect()
    print("Connected. Running inference episode against full benchmark workload...")
    buf, ep_reward, steps = run_episode(policy, frame, collect=False)
    print(f"Inference complete: steps={steps} cumulative_reward={ep_reward:.4f}")
    frame.close()


if __name__ == "__main__":
    main()
