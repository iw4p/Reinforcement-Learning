import random
from pettingzoo.classic import tictactoe_v3

def play_random_agent(agent, obs):
    """
    Play a random action for the given agent.

    Args:
        agent (str): The name of the agent ('player_1' or 'player_2').
        obs (dict): The current observation.

    Returns:
        int: The chosen action.
    """
    valid_actions = [i for i, mask in enumerate(obs['action_mask']) if mask]
    return random.choice(valid_actions)

def play_human_agent(agent, obs):
    """
    Get a valid action from the human player.

    Args:
        agent (str): The name of the agent ('player_1' or 'player_2').
        obs (dict): The current observation.

    Returns:
        int: The chosen action.
    """
    valid_actions = [i for i, mask in enumerate(obs['action_mask']) if mask]
    action = None
    while action not in valid_actions:
        try:
            action = int(input(f"Player {agent[-1]}, please enter your action ({', '.join(map(str, valid_actions))}): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
    return action

def play_game(env, player1_agent, player2_agent):
    """
    Play a game of Tic-Tac-Toe with the given agents.

    Args:
        env (tictactoe_v3.env): The Tic-Tac-Toe environment.
        player1_agent (callable): The function to get the action for player 1.
        player2_agent (callable): The function to get the action for player 2.
    """
    env.reset()
    agents = ['player_1', 'player_2']
    not_finished = True
    while not_finished:
        for agent in agents:
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                not_finished = False
                break
            action = player1_agent(agent, observation) if agent == 'player_1' else player2_agent(agent, observation)
            env.step(action)
    print('Game Over')
    print('Final rewards:', env.rewards)

def main():
    env = tictactoe_v3.env(render_mode='human')
    options = {
        '1': (play_random_agent, play_human_agent),
        '2': (play_human_agent, play_random_agent),
        '3': (play_human_agent, play_human_agent)
    }
    choice = input("Choose an option:\n1. Random vs Human\n2. Human vs Random\n3. Human vs Human\n")
    if choice in options:
        play_game(env, *options[choice])
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    main()