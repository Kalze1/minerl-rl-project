import minerl
import gym
import torch
import torch.optim as optim
from agents.ppo_agent import PPOAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/")
writer.add_scalar("Reward", total_reward, episode)

# Initialize environment and agent
env = gym.make("MineRLNavigateDense-v0")
obs_space = env.observation_space["pov"].shape[0]
action_space = env.action_space.n

agent = PPOAgent(obs_space, action_space)
optimizer = optim.Adam(agent.parameters(), lr=1e-4)

# Training loop
for episode in tqdm(range(1000)):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.FloatTensor(obs["pov"]).unsqueeze(0)
        action_probs, _ = agent(obs_tensor)
        action = torch.multinomial(action_probs, 1).item()

        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode}, Reward: {total_reward}")

env.close()
