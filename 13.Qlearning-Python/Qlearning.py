import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class QLearning:
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_dec=0.995,
                 batch_size=64, memory_size=10000, target_update_freq=100):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq

        if hasattr(env.action_space, 'n'):
            self.discrete = True
            self.action_shape = env.action_space.n
        else:
            self.discrete = False
            self.action_shape = env.action_space.shape[0]
        
        self.state_shape = env.observation_space.shape
        self.q_eval = self.build_model()
        self.q_target = clone_model(self.q_eval)
        self.q_target.set_weights(self.q_eval.get_weights())

    def build_model(self):
        model = Sequential([
            Dense(64, input_shape=self.state_shape, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_shape)
        ])
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            if self.discrete:
                return np.random.choice(self.action_shape)
            else:
                return np.random.uniform(-1, 1, size=self.action_shape)
        else:
            if self.discrete:
                return np.argmax(self.q_eval.predict(state.reshape(1, -1)))
            else:
                return self.q_eval.predict(state.reshape(1, -1))[0]

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            target = self.q_eval.predict(state.reshape(1, -1))
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.max(self.q_target.predict(next_state.reshape(1, -1)))
            states.append(state)
            targets.append(target)
        states = np.array(states).reshape(self.batch_size, *self.state_shape)
        targets = np.array(targets).reshape(self.batch_size, self.action_shape)
        self.q_eval.fit(states, targets, verbose=0, epochs=1)

    def train(self, episodes=1000):
        rewards_per_episode = []
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                self.experience_replay()
                state = next_state
                total_reward += reward
            
            rewards_per_episode.append(total_reward)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec
            
            if episode % self.target_update_freq == 0:
                self.q_target.set_weights(self.q_eval.get_weights())
            
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
        
        return rewards_per_episode

# Example usage:
# ql = QLearning(env)
# rewards = ql.train(episodes=1000)
# print("Training complete.")