from mlrose import NeuralNetwork
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    data_set = 'loan'  # Wine data set is 'wine' and is 1,600 rows, Loan data set is 'loan' and is 30,000 rows

    # Load data sets
    if data_set is 'wine':
        data_path = 'winequality-red.csv'
    else:
        data_path = 'UCI_Credit_Card.csv'

    # Load CSV data
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    # Apply the following three algorithms to train weights on a Neural Network from assignment 1.
    # create a Neural Network Learner trying out a variety of weights
    hidden_layers = [1, 100]  # one layer of 100 hidden nodes (same I used in assignment 1)
    hidden_layers_2 = [2, 10]
    hidden_layers_3 = [4, 16]
    learning_rate = 0.1
    activation = 'tanh'
    max_iters = 200

    # Randomized Hill Climbing
    clf_1 = NeuralNetwork(hidden_nodes=hidden_layers_2, activation=activation,
                        max_iters=max_iters, learning_rate=learning_rate, algorithm='random_hill_climb')
    # Simulated Annealing
    clf_2 = NeuralNetwork(hidden_nodes=hidden_layers_2, activation=activation,
                        max_iters=max_iters, learning_rate=learning_rate, algorithm='simulated_annealing')
    # Genetic algorithm
    pop_size = 200
    mutation_prob = 0.1
    clf_3 = NeuralNetwork(hidden_nodes=hidden_layers_2, activation=activation,
                        max_iters=max_iters, learning_rate=learning_rate, algorithm='genetic_alg', pop_size=pop_size,
                          mutation_prob=mutation_prob)

    # Train and algorithms
    start_1 = timer()
    clf_1.fit(X_train, y_train)
    end_1 = timer()
    train_time_1 = end_1 - start_1

    # Predict accuracy Scores
    y_pred_1 = clf_1.predict(X_test)
    acc_1 = accuracy_score(y_test, y_pred_1) * 100

    # Display final results of training and prediction
    print("Random Hill Climb:")
    print("Train Time: {:10.6f}s".format(train_time_1))
    print("Accuracy: {:3.4f}%".format(acc_1))

    start_2 = timer()
    clf_2.fit(X_train, y_train)
    end_2 = timer()
    train_time_2 = end_2 - start_2

    y_pred_2 = clf_2.predict(X_test)
    acc_2 = accuracy_score(y_test, y_pred_2) * 100

    print("Simulated Annealing")
    print("Train Time: {:10.6f}s".format(train_time_2))
    print("Accuracy: {:3.4f}%".format(acc_2))

    start_3 = timer()
    clf_3.fit(X_train, y_train)
    end_3 = timer()
    train_time_3 = end_3 - start_3

    y_pred_3 = clf_3.predict(X_test)
    acc_3 = accuracy_score(y_test, y_pred_3) * 100

    print("Genetic Algorithms")
    print("Train Time: {:10.6f}s".format(train_time_3))
    print("Accuracy: {:3.4f}%".format(acc_3))

    # From assignment 1, I used Grid searches to find the best parameters. For the sake of consistency, I will use the
    # exact same neural network in this assignment so I can directly compare the performance of the various algorithms to
    # back propagation.

    # Optimization Example Problems

    # One Max

    # Flip Flop

    # Four Peaks Problem (Simulated Annealing example)

    # K-Color Problem (MIMIC example)

    # Flip Flop with adjusted fitness function
