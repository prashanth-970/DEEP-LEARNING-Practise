import numpy as np
from perceptron import Perceptron
from environment import GridEnvironment

def load_trained_perceptron():
    perceptron = Perceptron(input_size=4)
    perceptron.weights = np.load("perceptron_weights.npy")
    perceptron.bias = np.load("perceptron_bias.npy")
    return perceptron

def run_simulation():
    env = GridEnvironment(size=10)
    perceptron = load_trained_perceptron()

    for step in range(50):
        state = env.get_state()
        action = np.argmax(perceptron.predict(state))  # Choose the best action
        reached_goal = env.move_agent(action)
        print(f"Step {step+1}: Agent at {env.agent_pos}")

        if reached_goal:
            print("Goal reached!")
            break

if __name__ == "__main__":
    run_simulation()
