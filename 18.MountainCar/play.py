import numpy as np
import gymnasium as gym

# load trained Q table
Q = np.load('MountainCar-v0-q-learning.npy')

# create an env
env = gym.make("MountainCar-v0", render_mode="human")

# seed 
# env.action_space.seed(42)

# observation, info = env.reset(seed=42)

observation, info = env.reset()

# total number of steps
for s in range(200):
    
    state = (observation - env.observation_space.low)*np.array([10, 100])
    state = np.round(state, 0).astype(int)
    
    # define the next step to take
    next_action = np.argmax(Q[state[0], state[1]]) 

    # perform one step
    observation, reward, terminated, truncated, info = env.step(next_action)
    print(s, observation, reward, terminated, truncated, info)

    # if the game ends, restart the game
    if terminated or truncated:
        observation, info = env.reset()