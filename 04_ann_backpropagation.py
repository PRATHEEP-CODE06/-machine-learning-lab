"""
Experiment 4: Artificial Neural Network with Backpropagation
Build an Artificial Neural Network by implementing the Backpropagation algorithm
from scratch and test it on a binary logic dataset (XOR).
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_neurons = 2
    hidden_neurons = 4
    output_neurons = 1
    learning_rate = 0.5
    epochs = 10000

    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        hidden_input = np.dot(X, W1) + b1
        hidden_output = sigmoid(hidden_input)

        final_input = np.dot(hidden_output, W2) + b2
        predicted_output = sigmoid(final_input)

        error = y - predicted_output
        d_output = error * sigmoid_derivative(predicted_output)

        error_hidden = d_output.dot(W2.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_output)

        W2 += hidden_output.T.dot(d_output) * learning_rate
        b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden) * learning_rate
        b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

        if (epoch + 1) % 2000 == 0:
            loss = np.mean(np.square(error))
            print(f"Epoch {epoch + 1}/{epochs} - MSE Loss: {loss:.6f}")

    print("\nFinal Predictions after Training:")
    for inp, target, pred in zip(X, y, predicted_output):
        print(f"Input: {inp} | Target: {target[0]} | Predicted: {pred[0]:.4f} (Class: {round(pred[0])})")
