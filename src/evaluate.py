from agents.ppo_agent import PPOAgent
import minerl
import gym
import torch

env = gym.make("MineRLNavigateDense-v0")
agent = PPOAgent(obs_space, action_space)
agent.load_state_dict(torch.load("models/ppo_agent.pth"))

obs = env.reset()
done = False
while not done:
    obs_tensor = torch.FloatTensor(obs["pov"]).unsqueeze(0)
    action_probs, _ = agent(obs_tensor)
    action = torch.multinomial(action_probs, 1).item()
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
