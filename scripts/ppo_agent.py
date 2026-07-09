"""
ppo_agent.py — PPO agent for NAND flash GC victim-block selection.

The action space is "pick one of N candidate blocks to erase". N varies
step to step, so we frame this as a scoring problem: the policy network
scores each candidate block from its per-block features (5 numbers) plus
shared global features (4 numbers), and we sample / argmax over a
softmax across the *available* candidates (variable-size action space,
handled by masking).

State per candidate: [invalid_ratio, valid_ratio, free_ratio, pe_norm, age_norm]
Global state (shared across candidates): [waf, util, free_ratio_global, gc_pressure]

Reward: shaped as negative WAF delta at each GC decision, i.e. agent is
rewarded for choices that keep the cumulative write-amplification low.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

GLOBAL_FEAT = 4
CAND_FEAT   = 5
MAX_CAND    = 64


class PPOPolicy(nn.Module):
    """Scores each candidate block; shares an MLP encoder across candidates."""

    def __init__(self, global_dim=GLOBAL_FEAT, cand_dim=CAND_FEAT, hidden=64):
        super().__init__()
        in_dim = global_dim + cand_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.score_head = nn.Linear(hidden, 1)       # per-candidate logit
        self.value_head = nn.Sequential(              # state value (global only)
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_feat, cand_feats, mask):
        """
        global_feat: (B, global_dim)
        cand_feats : (B, MAX_CAND, cand_dim)
        mask       : (B, MAX_CAND)  1 = valid candidate, 0 = padding
        returns logits (B, MAX_CAND), value (B,)
        """
        B, N, _ = cand_feats.shape
        g_exp = global_feat.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([g_exp, cand_feats], dim=-1)
        h = self.encoder(x)
        logits = self.score_head(h).squeeze(-1)             # (B, N)
        logits = logits.masked_fill(mask == 0, float("-1e9"))
        value = self.value_head(global_feat).squeeze(-1)    # (B,)
        return logits, value

    def act(self, global_feat, cand_feats, mask):
        with torch.no_grad():
            logits, value = self.forward(global_feat, cand_feats, mask)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action, logp, value


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.global_feats = []
        self.cand_feats   = []
        self.masks        = []
        self.actions      = []
        self.logps        = []
        self.values       = []
        self.rewards      = []
        self.dones        = []

    def add(self, g, c, m, a, lp, v, r, d):
        self.global_feats.append(g)
        self.cand_feats.append(c)
        self.masks.append(m)
        self.actions.append(a)
        self.logps.append(lp)
        self.values.append(v)
        self.rewards.append(r)
        self.dones.append(d)

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


class PPOTrainer:
    def __init__(self, policy, lr=3e-4, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, epochs=4, minibatch=64):
        self.policy   = policy
        self.optim    = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.vf_coef  = vf_coef
        self.ent_coef = ent_coef
        self.epochs   = epochs
        self.minibatch = minibatch

    def update(self, buf: RolloutBuffer):
        g  = torch.stack(buf.global_feats)
        c  = torch.stack(buf.cand_feats)
        m  = torch.stack(buf.masks)
        a  = torch.stack(buf.actions)
        old_logp = torch.stack(buf.logps)
        values   = np.array(buf.values, dtype=np.float32)
        rewards  = buf.rewards
        dones    = buf.dones

        adv, ret = compute_gae(rewards, values, dones)
        adv = torch.tensor(adv)
        ret = torch.tensor(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = len(buf)
        idxs = np.arange(N)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.minibatch):
                mb = idxs[start:start + self.minibatch]
                mb = torch.tensor(mb, dtype=torch.long)

                logits, value = self.policy(g[mb], c[mb], m[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(a[mb])
                entropy  = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(value, ret[mb])

                loss = policy_loss + self.vf_coef * value_loss \
                       - self.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"]  += value_loss.item()
                stats["entropy"]     += entropy.item()
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats


def featurize(frame_dict):
    """Convert raw IPC frame dict into (global_feat, cand_feats, mask) tensors."""
    feats = frame_dict["features"]
    n_cand = frame_dict["n_cand"]

    global_feat = np.array(feats[:GLOBAL_FEAT], dtype=np.float32)

    cand_feats = np.zeros((MAX_CAND, CAND_FEAT), dtype=np.float32)
    mask = np.zeros(MAX_CAND, dtype=np.float32)

    offset = GLOBAL_FEAT
    for i in range(min(n_cand, MAX_CAND)):
        start = offset + i * CAND_FEAT
        end = start + CAND_FEAT
        if end <= len(feats):
            cand_feats[i] = feats[start:end]
            mask[i] = 1.0

    return (
        torch.tensor(global_feat),
        torch.tensor(cand_feats),
        torch.tensor(mask),
    )
