import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human").env

state = env.reset()

env.step(1)
env.step(1)
env.step(1)
env.step(1)
env.step(1)
