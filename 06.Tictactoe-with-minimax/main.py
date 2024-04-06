from pettingzoo.classic import tictactoe_v3
import copy
import numpy as np

def play_random_agent(agent, obs):
    action = env.action_space(agent).sample()
    while obs['action_mask'][action] != 1:
        action = env.action_space(agent).sample()
    return action

def play_human_agent(agent, obs):
    action = int(input("Please enter your move (0-8): "))
    while action < 0 or action > 8 or obs['action_mask'][action] != 1:
        action = int(input("Invalid move! Please enter a valid move (0-8): "))
    return action

def evaluate_board(env_copy):
    observation, _, termination, _, _ = env_copy.last()
    if termination:
        rewards = env_copy.rewards
        if rewards['player_1'] == 1:
            return -1  # player_2 (minimizing player) won
        elif rewards['player_2'] == 1:
            return 1  # player_1 (maximizing player) won
        else:
            return 0  # draw
    return None

def play_min_max_agent(agent, obs, depth):
    valid_actions = np.where(obs['action_mask'] == 1)[0]
    best_score = float('-inf')
    best_move = None
    for action in valid_actions:
        if obs['action_mask'][action] == 1:
            env_copy = copy.deepcopy(env)
            env_copy.step(action)
            score = minimax(env_copy, depth, False)
            if score > best_score:
                best_score = score
                best_move = action
    return best_move

def minimax(env_copy, depth, maximizing_player):
    observation, _, termination, _, _ = env_copy.last()
    valid_actions = np.where(observation['action_mask'] == 1)[0]
    result = evaluate_board(env_copy)
    if result is not None:
        return result
    if maximizing_player:
        # Max
        max_eval = float('-inf')
        for action in valid_actions:
            if observation['action_mask'][action] == 1:
                env2 = copy.deepcopy(env_copy)
                env2.step(action)
                eval = minimax(env2, depth - 1, False)
                max_eval = max(max_eval, eval)
        return max_eval
    else:
        # Min
        min_eval = float('inf')
        for action in valid_actions:
            if observation['action_mask'][action] == 1:
                env2 = copy.deepcopy(env_copy)
                env2.step(action)
                eval = minimax(env2, depth - 1, True)
                min_eval = min(min_eval, eval)
        return min_eval

def play_game():
    global env
    env.reset()
    print("New Game Started!")
    print(env.render())
    while True:
        for agent in env.agents:
            observation, _, termination, _, _ = env.last()
            if termination:
                print(f'Rewards: {env.rewards}')
                return
            else:
                if agent == 'player_1':
                    action = play_human_agent(agent, observation)
                else:
                    action = play_min_max_agent(agent, observation, depth=difficulty_level)
                env.step(action)
                print(env.render())

def choose_difficulty():
    global difficulty_level
    difficulty_level = int(input("Choose difficulty level (1: Easy, 2: Medium, 3: Hard): "))
    while difficulty_level not in [1, 2, 3]:
        difficulty_level = int(input("Invalid difficulty level! Choose again (1: Easy, 2: Medium, 3: Hard): "))

if __name__ == "__main__":
    env = tictactoe_v3.env(render_mode='human')
    play_again = True
    while play_again:
        choose_difficulty()
        play_game()
        play_again_input = input("Do you want to play again? (yes/no): ").lower()
        play_again = play_again_input == 'yes'