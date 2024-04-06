import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Constants
AGENT = 1
PASSENGER = 4
GOAL = 8
BARRIER = -1
STREET = 0

# Class & Implementations
class Map:
    def __init__(self, length, height, barriers, agent_position, passenger_position, goal_position):
        self.length = length
        self.height = height
        self.map = np.zeros((self.height, self.length))
        self.barriers = barriers
        self.agent_position = agent_position
        self.passenger_position = passenger_position
        self.goal_position = goal_position

    def get_agent_position(self):
        return tuple(self.agent_position)

    def get_passenger_position(self):
        return tuple(self.passenger_position)

    def get_goal_position(self):
        return tuple(self.goal_position)

    def fill_map(self):
        for barrier in self.barriers:
            self.map[tuple(barrier)] = BARRIER

        self.map[self.get_agent_position()] = AGENT
        self.map[self.get_passenger_position()] = PASSENGER
        self.map[self.get_goal_position()] = GOAL

    def bfs_search(self, start, goal):
        frontier = deque([start])
        came_from = {}
        came_from[start] = None

        while frontier:
            current = frontier.popleft()

            if current == goal:
                break

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_x, next_y = current[0] + dx, current[1] + dy
                if 0 <= next_x < self.length and 0 <= next_y < self.height and self.map[next_y][next_x] != BARRIER:
                    next_pos = (next_x, next_y)
                    if next_pos not in came_from:
                        frontier.append(next_pos)
                        came_from[next_pos] = current

        if goal not in came_from:
            print("Error: No path found from", start, "to", goal)
            return []

        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def plot_map(self, path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='terrain', interpolation='nearest')

        if path:
            x, y = zip(*path)
            plt.plot(y, x, color='red', linewidth=2)

        plt.title('Map with Path')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.xticks(range(self.length))
        plt.yticks(range(self.height))
        plt.grid(visible=True)
        plt.show()

    def run(self):
        self.fill_map()
        start = self.get_agent_position()[::-1]
        passenger = self.get_passenger_position()[::-1]
        goal = self.get_goal_position()[::-1]

        path_to_passenger = self.bfs_search(start, passenger)
        path_to_goal = self.bfs_search(passenger, goal)

        if path_to_passenger and path_to_goal:
            print("Path to pick up passenger:", path_to_passenger)
            print("Path to leave at goal:", path_to_goal)

            self.plot_map(path_to_passenger + path_to_goal[1:])  # Concatenate paths to plot combined path
        else:
            print("Failed to find valid path from start to goal.")


if __name__ == "__main__":
    barriers = [
        (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
        (4, 2), (4, 3), (4, 5), (4, 6),
        (12, 4), (13, 4), (14, 4), (15, 4), (16, 4),
        (4, 12), (4, 13), (4, 14), (4, 15), (4, 16)
    ]

    map_instance = Map(20, 20, barriers, [0, 0], [10, 10], [19, 19])
    map_instance.run()