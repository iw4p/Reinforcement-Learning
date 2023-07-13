from pettingzoo.classic import tictactoe_v3
import copy
import numpy as np

env = tictactoe_v3.env(render_mode='human')
env.reset()

def play_random_agent(agent, obs):
    action = env.action_space(agent).sample()
    print(f'First action: {action}')
    print(f"Action mask: {obs['action_mask']}")
    while obs['action_mask'][action] != 1:
        action = env.action_space(agent).sample()
        print(f'selected action: {action}')
    return action

def play_human_agent(agent, obs):
    action = int(input("Please enter an action: "))
    while obs['action_mask'][action] != 1:
        action = int(input("Please enter an action"))
    return action

def evaluate_board(env_copy):
    observation, reward, termination, truncation, info = env_copy.last()
    if termination or truncation:
        print(f'Reward: {env_copy.rewards}')
        if env_copy.rewards['player_1'] == 1: # Player 1 has won
            return -1
        elif env_copy.rewards['player_2'] == 1: # Player 2 has won
            return 1
        else: # Draw
            return 0
    return None

def play_min_max_agent(agent, obs):
    global env
    valid_actions = np.where(observation['action_mask'] == 1)[0]
    best_score = float('-inf')
    best_move = None

    for action in valid_actions:
        if observation['action_mask'][action] == 1:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            score = minimax(env_copy, 0, False)
            if score > best_score:
                best_score = score
                best_move = action
    return best_move



def minimax(env_copy, depth, maximizing_player):
    observation, reward, termination, truncation, info = env_copy.last()
    valid_actions = np.where(observation['action_mask'] == 1)[0]

    result = evaluate_board(env_copy)
    if result is not None:
        return result
    
    if maximizing_player: # Max
        max_eval = float('-inf')
        for action in valid_actions:
            if observation['action_mask'][action] == 1:
                env2 = copy.deepcopy(env_copy)
                env_copy.step(action)
                eval = minimax(copy.deepcopy(env_copy), depth + 1, False)
                env_copy = env2
                max_eval = max(max_eval, eval)
        return max_eval
    else: # Min
        min_eval = float('inf')
        for action in valid_actions:
            if observation['action_mask'][action] == 1:
                env2 = copy.deepcopy(env_copy)
                env_copy.step(action)
                eval = minimax(copy.deepcopy(env_copy), depth + 1, True)
                env_copy = env2
                min_eval = min(min_eval, eval)
        return min_eval

while not evaluate_board(env):
    for agent in env.agents:
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                print(f'Rewards: {env.rewards}')
                exit()
            else:
                if agent == 'player_1':
                    action = play_human_agent(agent, observation)
                else:
                    action = play_min_max_agent(agent, observation)
                print('play: ', action)
                env.step(action)