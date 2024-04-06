import argparse
import gym
from stable_baselines3 import A2C, TD3, SAC
import os

log_dir = "logs"
model_dir = "models"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env, algorithm, timesteps):
    if algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
    else:
        print("Algorithm is not supported.")
        exit(1)
    
    i = 0
    while True:
        i += 1
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algorithm}_{timesteps * i}")

def test(env, algorithm, model_path):
    if algorithm == 'A2C':
        model = A2C.load(model_path, env=env)
    elif algorithm == 'TD3':
        model = TD3.load(model_path, env=env)
    elif algorithm == 'SAC':
        model = SAC.load(model_path, env=env)
    else:
        print("Algorithm not found.")
        exit(1)
    
    obs = env.reset()
    steps = 500
    while steps > 0:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            steps -= 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose Train or Test a Model using Stable Baselines3 and Gym environment")
    parser.add_argument("gymenv", help="Name of the Gym environment (e.g., Humanoid-v4, CartPole-v1)")
    parser.add_argument("algorithm", help="Choose an algorithm: A2C, TD3, SAC")
    parser.add_argument("-t", "--train", action='store_true', help="Train the model")
    parser.add_argument("-e", "--test", metavar='model_path', help="Test a pre-trained model")
    parser.add_argument("-s", "--steps", type=int, default=20000, help="Number of training steps (default: 20000)")
    args = parser.parse_args()

    if args.train:
        print("Training started...")
        gymenv = gym.make(args.gymenv)
        train(gymenv, args.algorithm, args.steps)
        print("Training completed.")
    
    if args.test:
        print("Testing started...")
        gymenv = gym.make(args.gymenv)
        test(gymenv, args.algorithm, model_path=args.test)
        print("Testing completed.")

'''
requirement : pip install 'shimmy>=0.2.1'
The correct usage of the script is as follows: python main.py <gymenv> <algorithm> [-t | -e <model_path>] [-s <steps>]


Additionally, you can provide optional arguments:

-t to train the model
-e <model_path> to test a pre-trained model
-s <steps> to specify the number of training steps (default is 20000)

For example, to train an A2C model on the CartPole-v1 environment with 50000 steps, you would run:
python main.py CartPole-v1 A2C -t -s 50000

Or to test a pre-trained model on the CartPole-v1 environment, you would run:
python main.py CartPole-v1 A2C -e path_to_model

Make sure to replace path_to_model with the actual path to your pre-trained model.

'''