import numpy as np
from enum import Enum
import pandas as pd

class Perceptron:
    def __init__(self, weights: np.ndarray, bias: float, is_categorical: bool, threshold):
        self.weights = weights
        self.bias = bias
        self.delta_weights = np.zeros_like(weights)
        self.delta_bias = 0
        self.activity = 0
        self.activation = 0
        self.output = None
        self.delta = 0
        self.is_categorical = is_categorical
        self.output_threshold = threshold

    # Calculate the sigmoid activation function
    def calc_sig_activation(self):
        self.activation = 1 / (1 + np.exp(-self.activity))

    # Calculate the activity of the perceptron
    def calc_activity(self, inputs):
        print(inputs)
        print(self.weights)
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

    def train(self, inputs, target, eta, use_bias):
        self.calc_activity(inputs)
        self.calc_sig_activation()
        error = target - self.activation
        self.calc_output_delta(error)
        self.set_delta_weights(inputs, eta, use_bias)
        self.update_weights()

    def predict(self, inputs):
        self.calc_activity(inputs)
        self.calc_sig_activation()
        if self.is_categorical:
            self.output = 1 if self.activation > self.output_threshold else 0
        else:
            self.output = self.activation

        print(f"Inputs: {inputs}")
        print(f"Activity Value: {self.activity}")
        print(f"Activation Value: {self.output} because the data is {self.is_categorical} categorical")


class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, bias, learning_rate, use_bias=False):
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
        self.hidden_perceptron_1 = Perceptron(self.hidden_layer_weights[0], self.bias, False, None)
        self.hidden_perceptron_2 = Perceptron(self.hidden_layer_weights[1], self.bias, False, None)
        self.output_perceptron = Perceptron(self.output_layer_weights[0], self.bias, True, 0)

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
    hidden_layer_weights = [np.random.rand(2), np.random.rand(2)]
    output_layer_weights = [np.random.rand(2)]
    bias = 0
    learning_rate = 1
    use_bias = True
    data = pd.read_csv('D:\\NeuralNetworkProject\\data.csv')
    print(data)

    #--------------------TRAIN WITH SINGLE PERCEPTRON---------------------------#
    weights = np.random.rand(2)
    bias = -0.5
    is_categorical = True
    threshold = 0

    perceptron = Perceptron(weights, bias, is_categorical, threshold)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(30):
        print(f"Iteration {i + 1} --------------------------------------------------")

        for j in range(len(data)):
            if j % 2 != 0:
                inputs = data.iloc[j, 1:3].values
                target = data.iloc[j, 3]
                print(f"Input/Output Pair {j}: {inputs}, {target}")
                perceptron.train(inputs, target, learning_rate, 1)
                if perceptron.output == target:
                    if target == 0:
                        tn += 1
                    else:
                        tp += 1
                else:
                    if target == 1:
                        fn += 1
                    else:
                        fp += 1


    print(f'weights after training {perceptron.weights}')
    print(f'tn {tn}; tp {tp}; fn {fn}; fp {fp}')
    if (fn+tn) > 0:
        print(f'specifity of current network is {tn/(fn+tn)}')
    if (fp+tp) > 0:
        print(f'sensitivity of current network is {tp/(fp+tp)}')
    if (tp+tn) > 0:
        print(f'positive predictive value is {tp/(tp+tn)}')
    if (fp+fn) > 0:
        print(f'negative predictive value is {fn/(fn+fp)}')


    #--------------------TRAIN WITH HIDDEN LAYER---------------------------#
    nn_2 = NeuralNetwork(hidden_layer_weights, output_layer_weights, bias, learning_rate, use_bias)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    print(f"Online training for odd numbered items --------------------------------------------------")
    for i in range(30):
        print(f"Iteration {i + 1} --------------------------------------------------")
        for j in range(len(data)+1):
            if j % 2 != 0:
                inputs = data.iloc[j, 1:3].values
                target = data.iloc[j, 3]
                print(f"Input/Output Pair {j}: {inputs}, {target} ==========================================")
                nn_2.train(inputs, target)
                if nn_2.output_perceptron.output == target:
                    if target == 0:
                        tn += 1
                    else:
                        tp += 1
                else:
                    if target == 1:
                        fn += 1
                    else:
                        fp += 1


    print(f'tn {tn}; tp {tp}; fn {fn}; fp {fp}')
    if (fn+tn) > 0:
        print(f'specifity of current network is {tn/(fn+tn)}')
    if (fp+tp) > 0:
        print(f'sensitivity of current network is {tp/(fp+tp)}')
    if (tp+tn) > 0:
        print(f'positive predictive value is {tp/(tp+tn)}')
    if (fp+fn) > 0:
        print(f'negative predictive value is {fn/(fn+fp)}')

    print(f'hidden layer weights after training: {nn_2.hidden_layer_weights}')
    print(f'outer layer weights after training: {nn_2.output_layer_weights}')

    # Determine threshold



if __name__ == "__main__":
    main()
