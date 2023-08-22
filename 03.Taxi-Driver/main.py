# Imports
import numpy as np
from collections import deque

# Constant
AGENT = 1
PASSENGER = 4
GOAL = 8
BARRIER = -1

# Class & Implementations
class Map:
    def __init__(self, length, heigth, barriers, agent, passenger, goal):
        self.length = length
        self.heigth = heigth
        self.map = np.zeros((self.heigth, self.length))
        self.barriers = barriers
        self.agent = agent
        self.passenger = passenger
        self.goal = goal

    def get_agent(self):
        return tuple(self.agent)
    
    def get_passenger(self):
        return tuple(self.passenger)
    
    def get_goal(self):
        return tuple(self.goal)
    
    def fill_map(self):
        if len(self.barriers) != 0:
            for barrier in self.barriers:
                self.map[tuple(barrier)] = BARRIER
        self.map[self.get_agent()] = AGENT
        self.map[self.get_passenger()] = PASSENGER
        self.map[self.get_goal()] = GOAL

    def search(self, start, end):
        queue = deque([[start]])
        seen = set([start])
        while queue:
            path = queue.popleft()
            x, y = path[-1]

            if self.map[x][y] == end:
                return path

            for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= x2 < self.length and 0 <= y2 < self.heigth and self.map[x2][y2] != BARRIER and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    def convert(self, path, action):
        out = []
        for i in range(1, len(path)):
            current = path[i-1]
            next = path[i]
            if (current[0] - next[0] == -1):
                out.append('right')
            if (current[0] - next[0] == 1):
                out.append('left')
            if (current[1] - next[1] == -1):
                out.append('down')
            if (current[1] - next[1] == 1):
                out.append('up')
        out.append(action)
        return out

    def find_passenger(self):
        path = self.search(self.get_agent()[::-1], PASSENGER)
        return self.convert(path, 'pick_up')

    def find_goal(self):
        path = self.search(self.get_passenger()[::-1], GOAL)
        return self.convert(path, 'leave')

    def run(self):
        self.fill_map()
        print(self.map)
        pick_passenger = self.find_passenger()
        leave_at_goal = self.find_goal()
        return (pick_passenger, leave_at_goal)

if __name__ == "__main__":
    map = Map(3, 3, [], [0,0], [1,1], [2,2])
    print(map.run())