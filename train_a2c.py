import argparse
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scopone import ScoponeEnv


# ------------------------------------------------------------- #
# A2C LSTM Policy
# ------------------------------------------------------------- #

class A2CLSTMPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, hidden=None):
        """
        obs_seq: [B, T, obs_dim]
        hidden: (h0, c0) each [1, B, hidden_dim]
        Returns:
          logits: [B, T, action_dim]
          values: [B, T]
          hidden_next: (h, c)
        """
        B, T, _ = obs_seq.shape
        x = self.fc(obs_seq)  # [B, T, hidden_dim]
        out, hidden_next = self.lstm(x, hidden)  # out: [B, T, hidden_dim]
        logits = self.policy_head(out)          # [B, T, action_dim]
        values = self.value_head(out).squeeze(-1)  # [B, T]
        return logits, values, hidden_next

    def initial_hidden(self, batch_size: int = 1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)


# ------------------------------------------------------------- #
# Training Loop
# ------------------------------------------------------------- #

def run_episode(
    env: gym.Env,
    policy: A2CLSTMPolicy,
    device: torch.device,
    gamma: float = 0.99,
) -> Tuple[float, int]:
    """
    Run one episode, collect transitions, do one A2C update.

    Returns: (episode_return, episode_length)
    """
    obs, _ = env.reset()
    done = False
    truncated = False

    obs_list = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    masks = []  # 0 if done, 1 otherwise

    hidden = policy.initial_hidden(batch_size=1, device=device)

    ep_return = 0.0
    ep_len = 0

    while not (done or truncated):
        obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,obs_dim]
        logits, value, hidden = policy(obs_tensor, hidden)
        logits = logits[:, -1, :]        # [1, action_dim]
        value = value[:, -1]             # [1]

        # Mask invalid actions (hand slots with no card) using obs structure:
        # hand_slot_mask is at index: we know obs structure in ScoponeEnv: last 183 dims,
        # but simplest: decode mask from obs: it's at positions 160..168
        hand_slot_mask = obs[160:160+9]  # 9 elements
        logit_np = logits.detach().cpu().numpy().reshape(-1)
        # set logits for invalid slots to a large negative
        for i in range(len(hand_slot_mask)):
            if hand_slot_mask[i] < 0.5:
                logit_np[i] = -1e9
        logits_masked = torch.from_numpy(logit_np).to(device).unsqueeze(0)

        dist = torch.distributions.Categorical(logits=logits_masked)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, truncated, info = env.step(int(action.item()))

        obs_list.append(obs_tensor.squeeze(0))  # [1, obs_dim]
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
        masks.append(torch.tensor([0.0 if (done or truncated) else 1.0],
                                  dtype=torch.float32, device=device))

        ep_return += reward
        ep_len += 1

        obs = next_obs

    # Convert to tensors
    obs_batch = torch.cat(obs_list, dim=0).unsqueeze(0)  # [1, T, obs_dim]
    actions_batch = torch.stack(actions).unsqueeze(0)    # [1, T]
    log_probs_batch = torch.stack(log_probs).unsqueeze(0)  # [1, T]
    values_batch = torch.stack(values).unsqueeze(0)      # [1, T]
    rewards_batch = torch.stack(rewards).unsqueeze(0)    # [1, T, 1] -> [1, T]
    rewards_batch = rewards_batch.squeeze(-1)
    masks_batch = torch.stack(masks).unsqueeze(0)        # [1, T, 1] -> [1, T]
    masks_batch = masks_batch.squeeze(-1)

    # Compute returns (reverse)
    T = rewards_batch.size(1)
    returns = torch.zeros_like(rewards_batch, device=device)
    R = torch.zeros(1, device=device)
    for t in reversed(range(T)):
        R = rewards_batch[0, t] + gamma * R * masks_batch[0, t]
        returns[0, t] = R

    advantages = returns - values_batch  # [1, T]

    return ep_return, ep_len, obs_batch, actions_batch, log_probs_batch, values_batch, returns, advantages


def train(
    total_episodes: int = 10000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    log_interval: int = 100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ScoponeEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = A2CLSTMPolicy(obs_dim, action_dim, hidden_dim=128).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    ep_returns = []

    for ep in range(1, total_episodes + 1):
        (
            ep_return,
            ep_len,
            obs_batch,
            actions_batch,
            log_probs_batch,
            values_batch,
            returns_batch,
            advantages_batch,
        ) = run_episode(env, policy, device, gamma=gamma)

        ep_returns.append(ep_return)

        advantages = advantages_batch.detach()
        log_probs = log_probs_batch
        values = values_batch
        returns = returns_batch

        # Policy loss: - E[log pi * A]
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = (returns - values).pow(2).mean()

        # Entropy
        with torch.no_grad():
            logits, _, _ = policy(obs_batch)
        logits = logits[0]  # [T, action_dim]
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if ep % log_interval == 0:
            avg_return = np.mean(ep_returns[-log_interval:])
            print(
                f"Episode {ep} | avg_return={avg_return:.3f} | "
                f"last_len={ep_len} | loss={loss.item():.3f}"
            )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    train(total_episodes=args.episodes)
