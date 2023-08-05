import gymnasium as gym
from Qlearning import QLearning
from Sarsa import Sarsa
import numpy as np
from numpy import loadtxt
from time import sleep

env = gym.make("CliffWalking-v0").env

# Qlearning
print("Training using Qlearning")
qlearning = QLearning(env, 0.4, 0.99, 0.7, 0.05, 0.99, 10000)
q_table, _ = qlearning.train('q-table-cliffwalking.csv')

# SARSA
print("Training using SARSA")
sarsa = Sarsa(env, 0.4, 0.99, 0.7, 0.05, 0.99, 10000)
sarsa_table, _ = sarsa.train('sarsa-table-cliffwalking.csv')

env = gym.make("CliffWalking-v0", render_mode="human").env

# QLearning
print("Playing using QLearning")
(state, _) = env.reset()
rewards_q = 0
actions_q = 0
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, truncated, _ = env.step(action)

    rewards_q = rewards_q + reward
    actions_q += 1
    sleep(1)
    

# Sarsa
print("Playing using Sarsa")
(state, _) = env.reset()
rewards_sarsa = 0
actions_sarsa = 0
done = False

while not done:
    action = np.argmax(sarsa_table[state])
    state, reward, done, truncated, _ = env.step(action)

    rewards_sarsa = rewards_sarsa + reward
    actions_sarsa += 1
    sleep(1)

# Result
print(f"Actions by Qlearning: {actions_q}")
print(f"Rewards for Qlearning: {rewards_q}")
print(f"Actions by SARSA: {actions_sarsa}")
print(f"Rewards for SARSA: {rewards_sarsa}")
