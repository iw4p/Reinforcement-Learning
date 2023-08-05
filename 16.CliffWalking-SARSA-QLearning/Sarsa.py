import numpy as np
import random
from numpy import savetxt

class Sarsa:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
    
    def select_action(self, state):
        randomNumber = random.uniform(0, 1)
        if randomNumber < self.epsilon:
            return self.env.action_space.sample() # Explore
        return np.argmax(self.q_table[state]) # Exploit
    
    def train(self, filename):
        actions_per_episode = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            reward = 0
            done = False
            actions = 0
            rewards = 0

            action = self.select_action(state)
            
            while not done:

                next_state, reward, done, truncated, terminal = self.env.step(action)

                old_value = self.q_table[state, action]
                next_action = self.select_action(next_state)
                
                new_value = old_value + self.alpha * (reward + self.gamma * self.q_table[next_state, next_action] - old_value)
                self.q_table[state, action] = new_value

                state = next_state
                action = next_action

                actions += 1
                rewards += reward

            actions_per_episode.append(rewards)
            if i % 100 == 0:
                print("Episodes: " + str(i) + "\n")

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        savetxt(filename, self.q_table, delimiter=",")
        return self.q_table, actions_per_episode