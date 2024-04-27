import numpy as np
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt

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
        # print(inputs)
        # print(self.weights)
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

    def predict(self, inputs, threshold):
        self.calc_activity(inputs)
        self.calc_sig_activation()
        if self.is_categorical:
            self.output = 1 if self.activation > threshold else 0
        else:
            self.output = self.activation

        # print(f"Inputs: {inputs}")
        # print(f"Activity Value: {self.activity}")
        # if self.is_categorical:
        #     print(f'Activation Value: {self.activation}')
        #     print(f"Output Value: {self.output} under {threshold}")
        return self.output


class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, bias, learning_rate, output_threshold, use_bias=False):
        self.input_values = None
        self.target = None
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.bias = bias
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.threshold = output_threshold

        self.little_error = None
        self.big_error = None

        # Initialize the perceptron layers of the neural network
        self.hidden_perceptron_1 = Perceptron(self.hidden_layer_weights[0], self.bias, False, None)
        self.hidden_perceptron_2 = Perceptron(self.hidden_layer_weights[1], self.bias, False, None)
        self.output_perceptron = Perceptron(self.output_layer_weights[0], self.bias, True, output_threshold)

    def feed_forward(self, threshold):
        self.hidden_perceptron_1.predict(self.input_values, threshold)
        self.hidden_perceptron_2.predict(self.input_values, threshold)
        pred = self.output_perceptron.predict(np.array(
            [self.hidden_perceptron_1.activation, self.hidden_perceptron_2.activation]),threshold)
        self.calculate_error()
        return pred

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
        # print(f"Little Error: {self.little_error}")
        self.big_error = 0.5 * (self.little_error ** 2)
        # print(f"Big Error: {self.big_error}")

    def train(self, inputs, target, threshold):
        self.input_values = inputs
        self.target = target
        self.feed_forward(threshold)
        self.back_propagation()
        # print(f"Target: {target}")
        # print(f"Output: {self.output_perceptron.activation}\n")

    def predict(self, inputs, threshold):
        self.input_values = inputs
        pred = self.feed_forward(threshold)
        return pred
        # print(f"Inputs: {inputs}")
        # print(f"Output: {self.output_perceptron.activation}\n")


def main():
    # Define the weights and bias for the neural network

    learning_rate = 1
    use_bias = True
    bias = -0.5
    data = pd.read_csv('D:\\NeuralNetworkProject\\data.csv')
    iteration = 10

    #--------------------TRAIN WITH SINGLE PERCEPTRON---------------------------#

    is_categorical = True
    thresholds = np.arange(0, 1, .01)

    tp_total = np.zeros(len(thresholds))
    tn_total = np.zeros(len(thresholds))
    fp_total = np.zeros(len(thresholds))
    fn_total = np.zeros(len(thresholds))

    test_tp_total = np.zeros(len(thresholds))
    test_tn_total = np.zeros(len(thresholds))
    test_fp_total = np.zeros(len(thresholds))
    test_fn_total = np.zeros(len(thresholds))

    # Loop over iterations
    for n in range(iteration):
        # Train the perceptron
        weights = np.random.rand(2)
        perceptron = Perceptron(weights, bias, is_categorical, 0.5)  # Assuming threshold is 0.5
        for _ in range(30):
            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    perceptron.train(inputs, target, learning_rate, 1)

        # Loop over thresholds
        for t, threshold in enumerate(thresholds):
            # Calculate TP, TN, FP, FN after training
            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = perceptron.predict(inputs, threshold)
                    if prediction == target:
                        if target == 0:
                            tn_total[t] += 1
                        else:
                            tp_total[t] += 1
                    else:
                        if target == 1:
                            fn_total[t] += 1
                        else:
                            fp_total[t] += 1
            for j in range(len(data)):
                if j % 2 != 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = perceptron.predict(inputs, threshold)
                    if prediction == target:
                        if target == 0:
                            test_tn_total[t] += 1
                        else:
                            test_tp_total[t] += 1
                    else:
                        if target == 1:
                            test_fn_total[t] += 1
                        else:
                            test_fp_total[t] += 1

    # Compute averages across iterations
    tp_avg = tp_total / iteration
    tn_avg = tn_total / iteration
    fp_avg = fp_total / iteration
    fn_avg = fn_total / iteration
    precision_train = tp_avg / (tp_avg + fp_avg)
    recall_train = tp_avg / (tp_avg + fn_avg)
    f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

    tp_avg_test = test_tp_total/iteration
    tn_avg_test = test_tn_total/iteration
    fn_avg_test = test_fn_total/iteration
    fp_avg_test = test_fp_total/iteration

    # Find the optimal threshold based on the F1 score
    optimal_threshold_neural_network = thresholds[np.argmax(f1_score_train)]
    print(f"Optimal threshold for single percetron neural network: {optimal_threshold_neural_network}")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots

    # Plot training data
    axs[0].plot(thresholds, tp_avg, label='True Positives')
    axs[0].plot(thresholds, tn_avg, label='True Negatives')
    axs[0].plot(thresholds, fp_avg, label='False Positives')
    axs[0].plot(thresholds, fn_avg, label='False Negatives')
    axs[0].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Single Perceptron')
    axs[0].set_ylabel('Count')
    axs[0].axvline(optimal_threshold_neural_network, color='r', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold_neural_network:.2f}')
    axs[0].legend()

    # Plot test data
    axs[1].plot(thresholds, tp_avg_test, label='True Positives')
    axs[1].plot(thresholds, tn_avg_test, label='True Negatives')
    axs[1].plot(thresholds, fp_avg_test, label='False Positives')
    axs[1].plot(thresholds, fn_avg_test, label='False Negatives')
    axs[1].set_title('Average TP/TN/FP/FN across Thresholds on Test Data by Single Perceptron')
    axs[1].set_xlabel('Thresholds')
    axs[1].set_ylabel('Count')
    axs[1].axvline(optimal_threshold_neural_network, color='r', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold_neural_network:.2f}')
    axs[1].legend()

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()

    # #--------------------TRAIN WITH HIDDEN LAYER---------------------------#

    tp_total = np.zeros(len(thresholds))
    tn_total = np.zeros(len(thresholds))
    fp_total = np.zeros(len(thresholds))
    fn_total = np.zeros(len(thresholds))
    print(f'tp total {tp_total}')

    test_tp_total = np.zeros(len(thresholds))
    test_tn_total = np.zeros(len(thresholds))
    test_fp_total = np.zeros(len(thresholds))
    test_fn_total = np.zeros(len(thresholds))
    print(f"Online training for odd numbered items --------------------------------------------------")
    for n in range(iteration):
        # Train the network
        hidden_layer_weights = [np.random.rand(2), np.random.rand(2)]
        output_layer_weights = [np.random.rand(2)]
        nn_2 = NeuralNetwork(hidden_layer_weights, output_layer_weights, bias, learning_rate, 0.5, use_bias)
        for _ in range(30):
            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    nn_2.train(inputs, target, 0.5)
        # Loop over thresholds
        for t, threshold in enumerate(thresholds):
            # Calculate TP, TN, FP, FN after training
            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = nn_2.predict(inputs, threshold)
                    if prediction == target:
                        if target == 0:
                            tn_total[t] += 1
                        else:
                            tp_total[t] += 1
                    else:
                        if target == 1:
                            fn_total[t] += 1
                        else:
                            fp_total[t] += 1
            for j in range(len(data)):
                if j % 2 != 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = perceptron.predict(inputs, threshold)
                    if prediction == target:
                        if target == 0:
                            test_tn_total[t] += 1
                        else:
                            test_tp_total[t] += 1
                    else:
                        if target == 1:
                            test_fn_total[t] += 1
                        else:
                            test_fp_total[t] += 1

                # Compute averages across iterations
    print(f'tp total after training {tp_total}')
    tp_avg = tp_total / iteration
    tn_avg = tn_total / iteration
    fp_avg = fp_total / iteration
    fn_avg = fn_total / iteration
    precision_train = tp_avg / (tp_avg + fp_avg)
    recall_train = tp_avg / (tp_avg + fn_avg)
    f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

    tp_avg_test = test_tp_total / iteration
    tn_avg_test = test_tn_total / iteration
    fn_avg_test = test_fn_total / iteration
    fp_avg_test = test_fp_total / iteration

    # Find the optimal threshold based on the F1 score
    optimal_threshold_neural_network = thresholds[np.argmax(f1_score_train)]
    print(f"Optimal threshold for hidden layer neural network: {optimal_threshold_neural_network}")

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots

    # Plot training data
    axs[0].plot(thresholds, tp_avg, label='True Positives')
    axs[0].plot(thresholds, tn_avg, label='True Negatives')
    axs[0].plot(thresholds, fp_avg, label='False Positives')
    axs[0].plot(thresholds, fn_avg, label='False Negatives')
    axs[0].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Hidden Layered Network')
    axs[0].set_ylabel('Count')
    axs[0].axvline(optimal_threshold_neural_network, color='r', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold_neural_network:.2f}')
    axs[0].legend()

    # Plot test data
    axs[1].plot(thresholds, tp_avg_test, label='True Positives')
    axs[1].plot(thresholds, tn_avg_test, label='True Negatives')
    axs[1].plot(thresholds, fp_avg_test, label='False Positives')
    axs[1].plot(thresholds, fn_avg_test, label='False Negatives')
    axs[1].set_title('Average TP/TN/FP/FN across Thresholds on Test Data by Hidden Layered Network')
    axs[1].set_xlabel('Thresholds')
    axs[1].set_ylabel('Count')
    axs[1].axvline(optimal_threshold_neural_network, color='r', linestyle='--',
                   label=f'Optimal Threshold: {optimal_threshold_neural_network:.2f}')
    axs[1].legend()

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()


if __name__ == "__main__":
    main()
