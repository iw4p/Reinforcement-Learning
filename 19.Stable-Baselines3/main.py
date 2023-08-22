import argparse
import gymnasium as gym
from stable_baselines3 import A2C, TD3, SAC
import os

log_dir = "logs"
model_dir = "models"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, algorithm):
    if algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print("Algorithm is not supported.")
        exit(1)
    
    TIMESTEPS = 20000
    i = 0
    while True:
        i += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS*i}")

def test(env, algorithm, model_path):
    if algorithm == 'A2C':
        model = A2C.load(model_path, env=env)
    elif algorithm == 'TD3':
        model = TD3.load(model_path, env=env)
    elif algorithm == 'SAC':
        model = SAC.load(model_path, env=env)
    else:
        print("Not found.")
        exit(1)
    
    obs = env.reset()[0]
    done = False
    steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            steps -= 1
            if steps < 0:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose Train or Test a Model using Stable Baselines3 and gymnasium env")
    parser.add_argument("gymenv", help="For example: Humanoid-v4, CartPole-v1")
    parser.add_argument("algorithm")
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-e", "--test", metavar='model_path', help="Choose an algo like A2C, SAC, TD3")
    args = parser.parse_args()

    if args.train:
        print("train started")
        gymenv = gym.make(args.gymenv, render_mode=None)
        train(gymenv, args.algorithm)
    
    if args.test:
        gymenv = gym.make(args.gymenv, render_mode='human')
        test(gymenv, args.algorithm, model_path=args.test)