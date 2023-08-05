import gymnasium as gym
from numpy import loadtxt
import numpy as np

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human").env
sarsa_table = loadtxt('sarsa-table3.csv', delimiter=',')

(state, _) = env.reset()
epochs = 0
rewards = 0
done = False
while not done:
    action = np.argmax(sarsa_table[state])
    state, reward, done, _, _ = env.step(action)
    rewards += reward
    epochs += 1
 
print(f"Epochs: {epochs}")
print(f"Rewards: {rewards}")