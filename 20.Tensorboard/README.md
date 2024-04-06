# Reinforcement Learning with Stable Baselines3 and Gym

This repository contains a Python script for training and testing reinforcement learning algorithms using Stable Baselines3 on Gym environments.

## Overview

The script allows you to train and test reinforcement learning algorithms such as A2C, TD3, and SAC on various Gym environments. It provides options to specify the Gym environment, algorithm, and whether to train or test a model.

## Installation

To use this script, you need to install the required dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```

Additionally, you need to install `shimmy`:

```bash
pip install 'shimmy>=0.2.1'
```

## Usage

To run the script, use the following command:

```bash
python main.py <gymenv> <algorithm> [-t | -e <model_path>] [-s <steps>]
```

Replace `<gymenv>` with the name of the Gym environment (e.g., `Humanoid-v4`, `CartPole-v1`) and `<algorithm>` with the algorithm you want to use (e.g., `A2C`, `TD3`, `SAC`).

You can provide optional arguments:
- `-t` to train the model
- `-e <model_path>` to test a pre-trained model
- `-s <steps>` to specify the number of training steps (default is 20000)

For example, to train an A2C model on the CartPole-v1 environment with 50000 steps:

```bash
python main.py CartPole-v1 A2C -t -s 50000
```

Or to test a pre-trained model on the CartPole-v1 environment:

```bash
python main.py CartPole-v1 A2C -e path_to_model
```

Make sure to replace `path_to_model` with the actual path to your pre-trained model.