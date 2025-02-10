import numpy as np 
import matplotlib.pyplot as plt
from environment import GridEnvironment

def visualize_environment(env):
    grid = np.zeros((env.size, env.size))

    for x, y in env.obstacles:
        grid[x, y] = -1  # Obstacles

    ax = plt.gca()
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    
    grid[env.agent_pos] = 1  # Agent
    grid[env.goal_pos] = 2  # Goal
    
    plt.imshow(grid, cmap="coolwarm", origin="upper")
    plt.title("Agent Navigation")
    plt.show()

if __name__ == "__main__":
    env = GridEnvironment(size=10)
    visualize_environment(env)
