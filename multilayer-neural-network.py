import numpy as np
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


class Perceptron:
    def __init__(self, weights: np.ndarray, bias: float, is_categorical: bool, threshold):
        self.big_error = None
        self.little_error = None
        self.target = None
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
        self.target = target
        error = target - self.activation
        self.calc_output_delta(error)
        self.set_delta_weights(inputs, eta, use_bias)
        self.update_weights()

    def calculate_error(self):
        self.little_error = self.target - self.activation
        # print(f"Little Error: {self.little_error}")
        self.big_error = 0.5 * (self.little_error ** 2)
        # print(f"Big Error: {self.big_error}")

    def predict(self, inputs, target, threshold):
        self.target = target
        self.calc_activity(inputs)
        self.calc_sig_activation()
        if self.is_categorical:
            self.output = 1 if self.activation > threshold else 0
        else:
            self.output = self.activation
        self.calculate_error()


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
        self.hidden_perceptron_1.predict(self.input_values, self.target, threshold)
        self.hidden_perceptron_2.predict(self.input_values, self.target, threshold)
        pred = self.output_perceptron.predict(np.array(
            [self.hidden_perceptron_1.activation, self.hidden_perceptron_2.activation]), self.target, threshold)
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

    def predict(self, inputs, target, threshold):
        self.target = target
        self.input_values = inputs
        pred = self.feed_forward(threshold)
        return pred


def f_beta_score(precision, recall, beta, epsilon=1e-9):
    numerator = (1 + beta**2) * precision * recall
    denominator = beta**2 * precision + recall
    if denominator == 0:
        if precision == 0 and recall == 0:
            return 0.0  # Handle the case where both precision and recall are 0
        else:
            return float('nan')  # Return NaN for an undefined case
    else:
        return numerator / (denominator + epsilon)

def plot_precision_recall_curve(thresholds, tp, fp, tn, fn, title):
    precisions = []
    recalls = []

    for i in range(len(thresholds)):
        if tp[i] + fp[i] > 0:
            precision = tp[i] / (tp[i] + fp[i])
        else:
            precision = 0

        if tp[i] + fn[i] > 0:
            recall = tp[i] / (tp[i] + fn[i])
        else:
            recall = 0

        precisions.append(precision)
        recalls.append(recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.show()




def main():
    data = pd.read_csv('D:\\NeuralNetworkProject\\data.csv')  # would need to change to suit your machine
    iteration = 30  # number of iterations for significance
    learning_rate = 1  # learning rate of the network
    beta = 0.3 # weighing of precision vs recall in determining optimal threshold.
               # The lower the value of beta, the more weight will be given to precision compared to recall.
    use_bias = True
    bias = -0.5
    thresholds = np.arange(0, 1, .01)


    # #--------------------TRAIN WITH SINGLE PERCEPTRON---------------------------#
    is_categorical = True

    optimal_thresholds_perceptron = []
    tp_avg_list = []
    tn_avg_list = []
    fp_avg_list = []
    fn_avg_list = []

    tp_avg_list_test = []
    tn_avg_list_test = []
    fp_avg_list_test = []
    fn_avg_list_test = []

    big_e_avg_list = []
    big_e_avg_list_test = []

    # # Loop over iterations
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
    #        best_threshold = None
        best_f1_score = 0
        tp_list, tn_list, fp_list, fn_list = [], [], [], []
        tp_list_test, tn_list_test, fp_list_test, fn_list_test = [], [], [], []
        big_e_list, big_e_list_test = [], []
        for t, threshold in enumerate(thresholds):
            tp, tn, fp, fn = 0, 0, 0, 0
            tp_test, tn_test, fp_test, fn_test = 0, 0, 0, 0
            big_e, big_e_test = 0, 0

            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = perceptron.predict(inputs, target, threshold)
                    big_e += perceptron.big_error
                    if prediction == target:
                        if target == 0:
                            tn += 1
                        else:
                            tp += 1
                    else:
                        if target == 1:
                            fn += 1
                        else:
                            fp += 1
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)

            big_e_list.append(big_e)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = f_beta_score(precision, recall, beta)

            if f1_score > best_f1_score:
                best_threshold = threshold
                best_f1_score = f1_score

            for j in range(len(data)):
                if j % 2 != 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = perceptron.predict(inputs, target, threshold)
                    big_e_test += perceptron.big_error
                    if prediction == target:
                        if target == 0:
                            tn_test += 1
                        else:
                            tp_test += 1
                    else:
                        if target == 1:
                            fn_test += 1
                        else:
                            fp_test += 1
            tp_list_test.append(tp_test)
            tn_list_test.append(tn_test)
            fp_list_test.append(fp_test)
            fn_list_test.append(fn_test)

            big_e_list_test.append(big_e_test)


        optimal_thresholds_perceptron.append(best_threshold)
        tp_avg_list.append(tp_list)
        tn_avg_list.append(tn_list)
        fp_avg_list.append(fp_list)
        fn_avg_list.append(fn_list)

        tp_avg_list_test.append(tp_list_test)
        tn_avg_list_test.append(tn_list_test)
        fp_avg_list_test.append(fp_list_test)
        fn_avg_list_test.append(fn_list_test)

        big_e_avg_list.append(big_e_list)
        big_e_avg_list_test.append(big_e_list_test)

    avg_optimal_threshold = sum(optimal_thresholds_perceptron) / len(optimal_thresholds_perceptron)
    print(f"Average optimal threshold for single perceptron network: {avg_optimal_threshold}")

    # Calculate the average TP, TN, FP, and FN across iterations
    tp_avg = [sum(x) / iteration for x in zip(*tp_avg_list)]
    tn_avg = [sum(x) / iteration for x in zip(*tn_avg_list)]
    fp_avg = [sum(x) / iteration for x in zip(*fp_avg_list)]
    fn_avg = [sum(x) / iteration for x in zip(*fn_avg_list)]

    tp_avg_test = [sum(x) / iteration for x in zip(*tp_avg_list_test)]
    tn_avg_test = [sum(x) / iteration for x in zip(*tn_avg_list_test)]
    fp_avg_test = [sum(x) / iteration for x in zip(*fp_avg_list_test)]
    fn_avg_test = [sum(x) / iteration for x in zip(*fn_avg_list_test)]

    big_e_avg = [sum(x) / iteration for x in zip(*big_e_avg_list)]
    big_e_avg_test = [sum(x) / iteration for x in zip(*big_e_avg_list_test)]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots

    # Plot training data
    axs[0].plot(thresholds, tp_avg, label='True Positives')
    axs[0].plot(thresholds, tn_avg, label='True Negatives')
    axs[0].plot(thresholds, fp_avg, label='False Positives')
    axs[0].plot(thresholds, fn_avg, label='False Negatives')
    axs[0].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Single Perceptron Network')
    axs[0].set_ylabel('Count')
    axs[0].axvline(avg_optimal_threshold, color='r', linestyle='--',
                   label=f'Avg. Optimal Threshold: {avg_optimal_threshold:.2f}')
    axs[0].legend()

    axs[1].plot(thresholds, tp_avg_test, label='True Positives')
    axs[1].plot(thresholds, tn_avg_test, label='True Negatives')
    axs[1].plot(thresholds, fp_avg_test, label='False Positives')
    axs[1].plot(thresholds, fn_avg_test, label='False Negatives')
    axs[1].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Single Perceptron Network')
    axs[1].set_ylabel('Count')
    axs[1].axvline(avg_optimal_threshold, color='r', linestyle='--',
                   label=f'Avg. Optimal Threshold: {avg_optimal_threshold:.2f}')
    axs[1].legend()

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots
    axs[0].plot(thresholds, big_e_avg, label='training data')
    axs[1].plot(thresholds, big_e_avg_test, label='testing data')
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()

    plot_precision_recall_curve(thresholds, tp_avg, fp_avg,
                                tn_avg, fn_avg, 'Avg. Precision-Recall for Training Data by Single Perceptron Network')
    plot_precision_recall_curve(thresholds, tp_avg_test, fp_avg_test,
                                tn_avg_test, fn_avg_test,
                                'Avg. Precision-Recall for Testing Data by Single Perceptron Network')

    # #--------------------TRAIN WITH HIDDEN LAYER---------------------------#

    optimal_thresholds_nn = []
    tp_avg_list = []
    tn_avg_list = []
    fp_avg_list = []
    fn_avg_list = []

    tp_avg_list_test = []
    tn_avg_list_test = []
    fp_avg_list_test = []
    fn_avg_list_test = []

    big_e_avg_list = []
    big_e_avg_list_test = []

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
        best_threshold = None
        best_f1_score = 0
        tp_list, tn_list, fp_list, fn_list = [], [], [], []
        tp_list_test, tn_list_test, fp_list_test, fn_list_test = [], [], [], []
        big_e_list, big_e_list_test = [], []
        for t, threshold in enumerate(thresholds):
            tp, tn, fp, fn = 0, 0, 0, 0
            tp_test, tn_test, fp_test, fn_test = 0, 0, 0, 0
            big_e, big_e_test = 0, 0

            for j in range(len(data)):
                if j % 2 == 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = nn_2.predict(inputs, target, threshold)
                    big_e += nn_2.big_error
                    if prediction == target:
                        if target == 0:
                            tn += 1
                        else:
                            tp += 1
                    else:
                        if target == 1:
                            fn += 1
                        else:
                            fp += 1
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)

            big_e_list.append(big_e)

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = f_beta_score(precision, recall, beta)

            # Update the best threshold if this one has a better F1-score
            if f1_score > best_f1_score:
                best_threshold = threshold
                best_f1_score = f1_score

            for j in range(len(data)):
                if j % 2 != 0:
                    inputs = data.iloc[j, 1:3].values
                    target = data.iloc[j, 3]
                    prediction = nn_2.predict(inputs, target, threshold)
                    big_e_test += nn_2.big_error
                    if prediction == target:
                        if target == 0:
                            tn_test += 1
                        else:
                            tp_test += 1
                    else:
                        if target == 1:
                            fn_test += 1
                        else:
                            fp_test += 1
            tp_list_test.append(tp_test)
            tn_list_test.append(tn_test)
            fp_list_test.append(fp_test)
            fn_list_test.append(fn_test)

            big_e_list_test.append(big_e_test)

        optimal_thresholds_nn.append(best_threshold)
        tp_avg_list.append(tp_list)
        tn_avg_list.append(tn_list)
        fp_avg_list.append(fp_list)
        fn_avg_list.append(fn_list)

        tp_avg_list_test.append(tp_list_test)
        tn_avg_list_test.append(tn_list_test)
        fp_avg_list_test.append(fp_list_test)
        fn_avg_list_test.append(fn_list_test)

        big_e_avg_list.append(big_e_list)
        big_e_avg_list_test.append(big_e_list_test)

    avg_optimal_threshold = sum(optimal_thresholds_nn) / len(optimal_thresholds_nn)
    print(f"Average optimal threshold for hidden layer network: {avg_optimal_threshold}")

    # Calculate the average TP, TN, FP, and FN across iterations
    tp_avg = [sum(x) / iteration for x in zip(*tp_avg_list)]
    tn_avg = [sum(x) / iteration for x in zip(*tn_avg_list)]
    fp_avg = [sum(x) / iteration for x in zip(*fp_avg_list)]
    fn_avg = [sum(x) / iteration for x in zip(*fn_avg_list)]

    tp_avg_test = [sum(x) / iteration for x in zip(*tp_avg_list_test)]
    tn_avg_test = [sum(x) / iteration for x in zip(*tn_avg_list_test)]
    fp_avg_test = [sum(x) / iteration for x in zip(*fp_avg_list_test)]
    fn_avg_test = [sum(x) / iteration for x in zip(*fn_avg_list_test)]

    big_e_avg = [sum(x) / iteration for x in zip(*big_e_avg_list)]
    big_e_avg_test = [sum(x) / iteration for x in zip(*big_e_avg_list_test)]


    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots

    # Plot training data
    axs[0].plot(thresholds, tp_avg, label='True Positives')
    axs[0].plot(thresholds, tn_avg, label='True Negatives')
    axs[0].plot(thresholds, fp_avg, label='False Positives')
    axs[0].plot(thresholds, fn_avg, label='False Negatives')
    axs[0].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Hidden Layered Network')
    axs[0].set_ylabel('Count')
    axs[0].axvline(avg_optimal_threshold, color='r', linestyle='--',
                   label=f'Avg. Optimal Threshold: {avg_optimal_threshold:.2f}')
    axs[0].legend()

    axs[1].plot(thresholds, tp_avg_test, label='True Positives')
    axs[1].plot(thresholds, tn_avg_test, label='True Negatives')
    axs[1].plot(thresholds, fp_avg_test, label='False Positives')
    axs[1].plot(thresholds, fn_avg_test, label='False Negatives')
    axs[1].set_title('Average TP/TN/FP/FN across Thresholds on Training Data by Hidden Layered Network')
    axs[1].set_ylabel('Count')
    axs[1].axvline(avg_optimal_threshold, color='r', linestyle='--',
                   label=f'Avg. Optimal Threshold: {avg_optimal_threshold:.2f}')
    axs[1].legend()

    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Create two vertically aligned subplots
    axs[0].plot(thresholds, big_e_avg, label='training data')
    axs[1].plot(thresholds, big_e_avg_test, label='testing data')
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical spacing between subplots
    plt.show()

    plot_precision_recall_curve(thresholds, tp_avg, fp_avg,
                                tn_avg, fn_avg, 'Avg. Precision-Recall for Training Data by Hidden Layered Network')
    plot_precision_recall_curve(thresholds, tp_avg_test, fp_avg_test,
                                tn_avg_test, fn_avg_test,
                                'Avg. Precision-Recall for Testing Data by Hidden Layered Network')

    #-----------------------------Report final weights-----------------------------------------#
    # single perceptron
    print(f'final weights for single perceptron: {perceptron.weights}')
    # hidden layered network
    print(f'final weights for hidden layered network: \n')
    print(f'hidden nodes {nn_2.hidden_layer_weights}; output node {nn_2.output_layer_weights}')


    #-----------------------------Comparsion between networks-----------------------------------------#
    # Perform the repeated measures t-test
    t_statistic, p_value = stats.ttest_rel(optimal_thresholds_nn, optimal_thresholds_perceptron)
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between the optimal threshold values of the two networks.")
    else:
        print("There is no significant difference between the optimal threshold values of the two networks.")


if __name__ == "__main__":
    main()
