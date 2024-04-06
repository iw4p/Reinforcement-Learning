<h1 align="center">
  <br>
  <a href="https://github.com/iw4p/Reinforcement-Learning"><img src="https://github.com/iw4p/Reinforcement-Learning/assets/89135083/7102344f-ad00-4348-bda5-ca35f42c3c95" alt="Output" width="200"></a>
  <br>
  Reinforcement Leaning Project | Taxi Driver
  <br>
</h1>

<b><h4 align="center">.:: Map Navigation Project ::.</h4></b>

This project implements a simple map navigation system using Breadth-First Search (BFS) algorithm. It simulates a scenario where an agent needs to pick up a passenger from a certain location and drop them off at a goal location, while avoiding barriers on the map.

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example](#example)
5. [Contributing](#contributing)
6. [License](#license)

## Introduction

The project consists of a Python script (`main.py`) that defines a `Map` class. This class represents a map with barriers, an agent, a passenger, and a goal. It utilizes BFS algorithm to find the shortest path from the agent's starting position to the passenger, and then from the passenger to the goal.

## Installation

To use this project, you need to have Python installed on your system. You can download Python from the [official website](https://www.python.org/downloads/).

Clone the repository to your local machine:

```bash
git clone https://github.com/iw4p/Reinforcement-Learning.git
```

Navigate to the project directory:

```bash
cd 03.Taxi-Driver
```

## Usage

You can run the project by executing the `main.py` script:

```bash
python main.py
```

The script will generate a map with paths plotted from the agent's starting position to the passenger and then to the goal.

## Example

Suppose you have a map with barriers defined and the agent starting at position `[0, 0]`, the passenger at position `[10, 10]`, and the goal at position `[19, 19]`. You can define these parameters and run the script to visualize the paths.

```python
barriers = [
    (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
    (4, 2), (4, 3), (4, 5), (4, 6),
    (12, 4), (13, 4), (14, 4), (15, 4), (16, 4),
    (4, 12), (4, 13), (4, 14), (4, 15), (4, 16)
]

map_instance = Map(20, 20, barriers, [0, 0], [10, 10], [19, 19])
map_instance.run()
```

This will display the map with the calculated paths to pick up the passenger and reach the goal.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request on the GitHub repository.