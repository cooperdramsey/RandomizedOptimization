from mlrose.mlrose import NeuralNetwork
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def load_data(data_path):
    data = pd.read_csv(data_path)
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    return X, y


def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def plot_time_curve(title, iterations, training_times):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Training Time (s)")
    plt.grid()
    plt.plot(iterations, training_times, 'o-', color="r")

    return plt


def plot_accuracy_curve(title, iterations, accuracies):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(iterations, accuracies, 'o-', color="r")

    return plt


if __name__ == '__main__':
    data_path = 'UCI_Credit_Card.csv'

    # Load CSV data
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    # Apply the following three algorithms to train weights on a Neural Network from assignment 1.
    # create a Neural Network Learner trying out a variety of weights
    hidden_layers = [2]  # one layer of 100 hidden nodes (same I used in assignment 1)
    learning_rate = 0.001
    activation = 'tanh'

    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    train_times = []
    accuracies = []
    for max_iters in iterations:
        # Randomized Hill Climbing
        total_runs = range(10)
        best_acc = 0
        best_time = float("inf")
        clf_1 = NeuralNetwork(hidden_nodes=hidden_layers, activation=activation,
                              max_iters=max_iters, learning_rate=learning_rate, algorithm='random_hill_climb')

        # Run each algorithm multiple times on the same settings to avoid local optima in results graphs
        for run in total_runs:
            # Train and algorithms
            start_1 = timer()
            clf_1.fit(X_train, y_train)
            end_1 = timer()
            train_time_1 = end_1 - start_1

            # Predict accuracy Scores
            y_pred_1 = clf_1.predict(X_test)
            acc_1 = accuracy_score(y_test, y_pred_1)

            if train_time_1 < best_time:
                best_time = train_time_1
            if acc_1 > best_acc:
                best_acc = acc_1

        train_times.append(best_time)
        accuracies.append(best_acc)

    # Plot Random hill climb results
    time_plot_1 = plot_time_curve("Random Hill Climb", iterations, train_times)
    acc_plot_1 = plot_accuracy_curve("Random Hill Climb", iterations, accuracies)

    # SIMULATED ANNEALING ########################################################
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    train_times = []
    accuracies = []
    for max_iters in iterations:
        # Randomized Hill Climbing
        total_runs = range(10)
        best_acc = 0
        best_time = float("inf")
        clf_2 = NeuralNetwork(hidden_nodes=hidden_layers, activation=activation,
                              max_iters=max_iters, learning_rate=learning_rate, algorithm='simulated_annealing')

        # Run each algorithm multiple times on the same settings to avoid local optima in results graphs
        for run in total_runs:
            # Train and algorithms
            start_2 = timer()
            clf_2.fit(X_train, y_train)
            end_2 = timer()
            train_time_2 = end_2 - start_2

            # Predict accuracy Scores
            y_pred_2 = clf_2.predict(X_test)
            acc = accuracy_score(y_test, y_pred_2)

            if train_time_2 < best_time:
                best_time = train_time_2
            if acc > best_acc:
                best_acc = acc

        train_times.append(best_time)
        accuracies.append(best_acc)

    # Plot Random hill climb results
    time_plot_2 = plot_time_curve("Simulated Annealing", iterations, train_times)
    acc_plot_2 = plot_accuracy_curve("Simulated Annealing", iterations, accuracies)

    # iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
    # train_times = []
    # accuracies = []
    # for max_iters in iterations:
    #     # Genetic algorithm
    #     pop_size = 200
    #     mutation_prob = 0.1
    #     clf_3 = NeuralNetwork(hidden_nodes=hidden_layers, activation=activation,
    #                           max_iters=max_iters, learning_rate=learning_rate, algorithm='genetic_alg',
    #                           pop_size=pop_size,
    #                           mutation_prob=mutation_prob, clip_max=5)
    #
    #     # Train and algorithms
    #     start_3 = timer()
    #     clf_3.fit(X_train, y_train)
    #     end_3 = timer()
    #     train_time_3 = end_3 - start_3
    #
    #     train_times.append(train_time_3)
    #
    #     y_pred_3 = clf_3.predict(X_test)
    #     acc_3 = accuracy_score(y_test, y_pred_3)
    #
    #     accuracies.append(acc_3)
    #
    # # Plot Random hill climb results
    # time_plot_3 = plot_time_curve("Genetic Algorithm", iterations, train_times)
    # acc_plot_3 = plot_accuracy_curve("Genetic Algorithm", iterations, accuracies)

    time_plot_1.show()
    acc_plot_1.show()
    time_plot_2.show()
    acc_plot_2.show()
    # time_plot_3.show()
    # acc_plot_3.show()

    # Optimization Example Problems

    # One Max

    # Flip Flop

    # Four Peaks Problem (Simulated Annealing example)

    # K-Color Problem (MIMIC example)

    # Flip Flop with adjusted fitness function
