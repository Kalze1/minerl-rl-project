import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value
