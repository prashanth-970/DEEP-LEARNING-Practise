import numpy as np

class GridEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = (np.random.randint(size), np.random.randint(size))
        self.goal_pos = (np.random.randint(size), np.random.randint(size))
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.size // 2):
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) != self.goal_pos and (x, y) != self.agent_pos:
                obstacles.append((x, y))
                self.grid[x, y] = -1  # Mark obstacles
        return obstacles

    def get_state(self):
        """ Compute sensor data (distance to goal and nearest obstacle) """
        dx, dy = np.array(self.goal_pos) - np.array(self.agent_pos)
        nearest_obstacle = min(
            self.obstacles, key=lambda obs: np.linalg.norm(np.array(obs) - np.array(self.agent_pos))
        )
        ox, oy = np.array(nearest_obstacle) - np.array(self.agent_pos)
        return np.array([dx, dy, ox, oy]) / self.size  # Normalize inputs

    def move_agent(self, action):
        """ Move agent based on perceptron decision """
        x, y = self.agent_pos
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)
        
        if (x, y) not in self.obstacles:
            self.agent_pos = (x, y)
        
        return self.agent_pos == self.goal_pos  # Return True if goal reached
