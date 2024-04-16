import numpy as np
from enum import Enum


# Perceptron Class, used to create a perceptron object within a neural network

class Perceptron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = weights
        self.bias = bias
        self.delta_weights = np.zeros_like(weights)
        self.delta_bias = 0
        self.activity = 0
        self.activation = 0
        self.delta = 0

    # Calculate the sigmoid activation function
    def calc_sig_activation(self):
        self.activation = 1 / (1 + np.exp(-self.activity))

    # Calculate the activity of the perceptron
    def calc_activity(self, inputs):
        self.activity = np.dot(inputs, self.weights) + self.bias

    def calc_hidden_delta(self, next_layer_delta, next_layer_weights):
        self.delta = (1 - self.activation) * self.activation * (next_layer_weights * next_layer_delta)

    def calc_output_delta(self, little_error):
        self.delta = self.activation * (1 - self.activation) * little_error

    # Calculate the change in weights and bias
    def set_delta_weights(self, inputs, eta, use_bias):
        self.delta_weights = eta * self.delta * inputs
        if use_bias:
            self.delta_bias = eta * self.delta

    # Apply the change in weights and bias
    def update_weights(self):
        self.weights += self.delta_weights
        self.bias += self.delta_bias

    def predict(self, inputs):
        self.calc_activity(inputs)
        self.calc_sig_activation()
        print(f"Inputs: {inputs}")

        print(f"Activity Value: {self.activity}")
        print(f"Activation Value: {self.activation}")


class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, bias, learning_rate, use_bias=False):
        self.input_values = None
        self.input_values = None
        self.target = None
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.bias = bias
        self.use_bias = use_bias
        self.learning_rate = learning_rate

        self.little_error = None
        self.big_error = None

        # Initialize the perceptron layers of the neural network
        self.hidden_perceptron_1 = Perceptron(self.hidden_layer_weights[0], self.bias)
        self.hidden_perceptron_2 = Perceptron(self.hidden_layer_weights[1], self.bias)
        self.output_perceptron = Perceptron(self.output_layer_weights[0], self.bias)

    def feed_forward(self):
        self.hidden_perceptron_1.predict(self.input_values)
        self.hidden_perceptron_2.predict(self.input_values)
        self.output_perceptron.predict(np.array(
            [self.hidden_perceptron_1.activation, self.hidden_perceptron_2.activation]))
        self.calculate_error()

    def back_propagation(self):
        self.output_perceptron.calc_output_delta(self.little_error)
        self.output_perceptron.set_delta_weights(
            np.array([self.hidden_perceptron_1.activation, self.hidden_perceptron_2.activation]), self.learning_rate,
            self.use_bias)

        self.hidden_perceptron_1.calc_hidden_delta(self.output_perceptron.delta, self.output_perceptron.weights[0])
        self.hidden_perceptron_2.calc_hidden_delta(self.output_perceptron.delta, self.output_perceptron.weights[1])

        self.hidden_perceptron_1.set_delta_weights(self.input_values, self.learning_rate, self.use_bias)
        self.hidden_perceptron_2.set_delta_weights(self.input_values, self.learning_rate, self.use_bias)

        self.output_perceptron.update_weights()
        self.hidden_perceptron_1.update_weights()
        self.hidden_perceptron_2.update_weights()

        self.output_layer_weights = [self.output_perceptron.weights]
        self.hidden_layer_weights = [self.hidden_perceptron_1.weights, self.hidden_perceptron_2.weights]

    def calculate_error(self):
        self.little_error = self.target - self.output_perceptron.activation
        print(f"Little Error: {self.little_error}")
        self.big_error = 0.5 * (self.little_error ** 2)
        print(f"Big Error: {self.big_error}")

    def train(self, inputs, target):
        self.input_values = inputs
        self.target = target
        self.feed_forward()
        self.back_propagation()
        print(f"Target: {target}")
        print(f"Output: {self.output_perceptron.activation}\n")

    def predict(self, inputs):
        self.input_values = inputs
        self.feed_forward()
        print(f"Inputs: {inputs}")
        print(f"Output: {self.output_perceptron.activation}\n")


def main():
    # Define the weights and bias for the neural network
    hidden_layer_weights = [np.array([0.3, 0.3]), np.array([0.3, 0.3])]
    output_layer_weights = [np.array([0.8, 0.8])]
    bias = 0
    learning_rate = 1
    use_bias = True

    class Method(Enum):
        Test = 0
        Method1 = 1
        Method2 = 2

    method = Method.Method2.value

    input_output_pair = [
        #
        (np.array([1, 1]), 0.9),
        (np.array([-1, -1]), 0.05)
    ]

    # Create a neural network with the defined weights and bias
    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, bias, learning_rate, use_bias)

    if method == 0:
        # Test
        print(f"Test --------------------------------------------------")
        input_output_pair = [
            (np.array([1, 2]), 0.7)
        ]
        nn.use_bias = False
        for i in range(2):
            print(f"Iteration {i + 1} --------------------------------------------------")
            for iop_index, (inputs, target) in enumerate(input_output_pair):
                print(
                    f"Input/Output Pair {iop_index + 1}: {inputs}, {target} ==========================================")
                nn.train(inputs, target)

    elif method == 1:
        # Method 1
        print(f"Method 1 --------------------------------------------------")
        for i in range(15):
            print(f"Iteration {i + 1} --------------------------------------------------")
            for iop_index, (inputs, target) in enumerate(input_output_pair):
                print(
                    f"Input/Output Pair {iop_index + 1}: {inputs}, {target} ==========================================")
                nn.train(inputs, target)
        for inputs, target in input_output_pair:
            print(f"Final Output --------------------------------------------------")
            print(f"Input: {inputs}, Target: {target}")
            nn.target = target
            nn.predict(inputs)
    elif method == 2:
        # Method 2
        print(f"Method 2 --------------------------------------------------")
        for inputs, target in input_output_pair:
            for i in range(15):
                print(f"Iteration {i + 1} --------------------------------------------------")
                nn.train(inputs, target)
        for inputs, target in input_output_pair:
            print(f"Final Output --------------------------------------------------")
            print(f"Input: {inputs}, Target: {target}")
            nn.target = target
            nn.predict(inputs)


if __name__ == "__main__":
    main()
