# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_ratio, update_epochs):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_epochs = update_epochs

    def compute_loss(self, state, action, old_log_prob, advantage):
        action_probs, state_value = self.model(state)
        dist = torch.distributions.Categorical(action_probs)
        new_log_prob = dist.log_prob(action)
        ratio = (new_log_prob - old_log_prob).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        return -(policy_reward + 0.5 * nn.functional.mse_loss(state_value, advantage)).mean()

    def update(self, rollouts):
        for _ in range(self.update_epochs):
            for state, action, old_log_prob, advantage in rollouts:
                loss = self.compute_loss(state, action, old_log_prob, advantage)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()