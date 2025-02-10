import numpy as np
from perceptron import Perceptron
from environment import GridEnvironment

def generate_training_data(env, samples=500):
    """ Generate training data with expert moves (shortest path strategy) """
    inputs, targets = [], []
    
    for _ in range(samples):
        env.agent_pos = (np.random.randint(env.size), np.random.randint(env.size))
        state = env.get_state()

        # Define optimal movement (expert labels)
        dx, dy, _, _ = state * env.size  # Denormalize distances
        target = np.array([0, 0, 0, 0])  # One-hot encoding of movement

        if abs(dx) > abs(dy):
            target[2 if dx < 0 else 3] = 1  # Left or Right
        else:
            target[0 if dy < 0 else 1] = 1  # Up or Down

        inputs.append(state)
        targets.append(target)

    return np.array(inputs), np.array(targets)

def train_perceptron():
    env = GridEnvironment(size=10)
    perceptron = Perceptron(input_size=4, learning_rate=0.1)
    inputs, targets = generate_training_data(env)

    for epoch in range(100):
        for x, y in zip(inputs, targets):
            perceptron.train(x, y)

    print("Training complete. Saving model...")
    np.save("perceptron_weights.npy", perceptron.weights)
    np.save("perceptron_bias.npy", perceptron.bias)

if __name__ == "__main__":
    train_perceptron()
