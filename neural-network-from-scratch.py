import numpy as np
import matplotlib.pyplot as plt
import gzip
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        n = len(training_data)
        test_accuracy = []

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                accuracy = self.evaluate(test_data)
                test_accuracy.append(accuracy)
                print(f"Epoch {j}: {accuracy} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")

        return test_accuracy

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def load_data(file):
    with gzip.open(file, 'rt') as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        labels = data[:, 0].astype(int)
        images = data[:, 1:] / 255.0
        images = [np.reshape(x, (784, 1)) for x in images]
        labels = [vectorized_result(y) for y in labels]
        return list(zip(images, labels))

def load_data_wrapper(train_file, test_file):
    training_data = load_data(train_file)
    test_data = load_data(test_file)
    return training_data, test_data

def main():
    if len(sys.argv) != 6:
        sys.exit(1)

    NInput = int(sys.argv[1])
    NHidden = int(sys.argv[2])
    NOutput = int(sys.argv[3])
    train_file = sys.argv[4]
    test_file = sys.argv[5]

    training_data, test_data = load_data_wrapper(train_file, test_file)

    # Task 1: Training with epoch = 30, mini-batch size = 20, eta = 3.0
    net = NeuralNetwork([NInput, NHidden, NOutput])
    test_accuracy1 = net.gradient_descent(training_data, epochs=30, mini_batch_size=20, eta=3.0, test_data=test_data)
    max_accuracy_task1 = max(test_accuracy1)
    plt.plot(test_accuracy1, label="eta=3.0")
    print(f"Maximum test accuracy (Task 1): {max_accuracy_task1}")

    # Task 2: Training with different learning rates
    learning_rates = [0.001, 0.01, 1.0, 10, 100]
    max_accuracies = []
    for eta in learning_rates:
        net = NeuralNetwork([NInput, NHidden, NOutput])
        test_accuracy = net.gradient_descent(training_data, epochs=30, mini_batch_size=20, eta=eta, test_data=test_data)
        max_accuracies.append(max(test_accuracy))
        plt.plot(test_accuracy, label=f"eta={eta}")

    # Task 3: Training with different mini-batch sizes
    batch_sizes = [1, 5, 20, 100, 300]
    max_accuracies_batch = []
    for mini_batch_size in batch_sizes:
        net = NeuralNetwork([NInput, NHidden, NOutput])
        test_accuracy = net.gradient_descent(training_data, epochs=30, mini_batch_size=mini_batch_size, eta=3.0, test_data=test_data)
        max_accuracies_batch.append(max(test_accuracy))

    plt.figure()
    plt.plot(batch_sizes, max_accuracies_batch, marker='o')
    plt.xlabel('Mini-batch Size')
    plt.ylabel('Maximum Test Accuracy')
    plt.title('Maximum Test Accuracy vs. Mini-batch Size')

    # Task 4: Different hyperparameter settings
    hyperparameters = [(epochs, mini_batch_size) for epochs in [10, 20, 50] for mini_batch_size in [10, 50, 100]]
    max_accuracies_hyper = []
    for epochs, mini_batch_size in hyperparameters:
        net = NeuralNetwork([NInput, NHidden, NOutput])
        test_accuracy = net.gradient_descent(training_data, epochs=epochs, mini_batch_size=mini_batch_size, eta=3.0, test_data=test_data)
        max_accuracies_hyper.append((epochs, mini_batch_size, max(test_accuracy)))

    for epochs, mini_batch_size, max_accuracy in max_accuracies_hyper:
        print(f"Maximum test accuracy for epochs={epochs}, mini-batch size={mini_batch_size}: {max_accuracy}")

    plt.legend()
    plt.show()
    for eta, max_accuracy in zip(learning_rates, max_accuracies):
        print(f"Maximum test accuracy for eta={eta}: {max_accuracy}")

if __name__ == "__main__":
    main()
