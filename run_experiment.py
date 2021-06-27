import argparse
import gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# note: 'FrozenLake-v0' is an example environment
parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v0")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    print(args.env +':Env')
    gym.envs.register(
        id=args.env + "-v0",
        entry_point=args.env +':Env',
    )
    env = gym.make(args.env + "-v0")
    print("Loaded", args.env)

action_dim = env.action_space.n
state_dim = env.observation_space.n

gamma = 0.9
agent = agentfile.Agent(state_dim, action_dim, gamma)

# average state-action values and rewards over five runs
iter = 10000
Q = np.zeros((state_dim, action_dim))
V = np.zeros(state_dim)
rewards = np.zeros((5,iter))
total_reward = 0
for i in range(5):
    observation = env.reset()
    for j in range(iter): 
        #env.render()
        action = agent.act(observation) 
        observation, reward, done, info = env.step(action)
        
        rewards[i, j] = reward
        total_reward += reward
        
        agent.observe(observation, reward, done)

        if done:
            observation = env.reset() 
    env.close()
    
    Q += ((agent.Q1 + agent.Q2)/2 - Q) / (i+1)

print(Q)
print(total_reward/i)

# plot moving average reward
window = 200
rewardsAvg = np.zeros((5, iter))
for i in range(5):
    for j in range(iter):
        start = max(0, j-window)
        rewardsAvg[i, j] = np.mean(rewards[i,start:j])

# plot 95 % confidence interval for moving average reward
mean = np.mean(rewardsAvg, axis=0)
std = np.std(rewardsAvg, axis=0)
t = 2.776
upper = mean + t * std / np.sqrt(5)
lower = mean - t * std / np.sqrt(5)
fig = plt.figure()
plt.xscale('log')
plt.plot(range(iter), np.mean(rewardsAvg, axis=0))
plt.fill_between(range(iter), upper, lower,  color='b', alpha=.05)
plt.xlabel('Run')
plt.ylabel('Moving Average Reward')
plt.show()