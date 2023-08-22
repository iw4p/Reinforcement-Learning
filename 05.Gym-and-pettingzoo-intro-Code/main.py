from pettingzoo.classic import tictactoe_v3

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


not_finish = True
while not_finish:
    for agent in ['player_1', 'player_2']:
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                not_finish = False
            else:
                if agent == 'player_1':
                    action = play_random_agent(agent, observation)
                else:
                    action = play_human_agent(agent, observation)
                print('play: ', action)
                env.step(action)
print(env.rewards)