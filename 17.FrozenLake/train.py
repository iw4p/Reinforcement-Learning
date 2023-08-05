import gymnasium as gym
from Sarsa import Sarsa
from utils import general_specific_plot, general_plot

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True).env

sarsa1 = Sarsa(env, 0.4, 0.95, 0.95, 0.0001, 0.9999, 20000)
sarsa_table1, r1 = sarsa1.train('sarsa-table1.csv')

sarsa2 = Sarsa(env, 0.2, 0.95, 0.95, 0.0001, 0.9999, 20000)
sarsa_table2, r2 = sarsa2.train('sarsa-table2.csv')

sarsa3 = Sarsa(env, 0.1, 0.95, 0.95, 0.0001, 0.9999, 20000)
sarsa_table3, r3 = sarsa3.train('sarsa-table3.csv')

sarsa4 = Sarsa(env, 0.05, 0.95, 0.95, 0.0001, 0.9999, 20000)
sarsa_table4, r4 = sarsa4.train('sarsa-table4.csv')

general_plot(r1, r2, r3, r4)
general_specific_plot(r1, r2, r3, r4)