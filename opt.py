from mlrose.mlrose import OneMax, FlipFlop, FourPeaks, CustomFitness, Queens, Knapsack
from mlrose.mlrose import random_hill_climb, simulated_annealing, genetic_alg, mimic, DiscreteOpt
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt


# Define the custom fitness function for MIMIC
def custom_fn(state):
    s = ""
    for i in range(100000):
        s += "1"

    return np.sum(state)


def plot_curve(title, xlabel, ylabel, x_data, y_data):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.plot(x_data, y_data, 'o-', color="r")

    return plt


def solve_hill(problem, problem_name):
    start = timer()
    best_state, best_fitness, fitness_curve = random_hill_climb(problem, curve=True, restarts=10)
    end = timer()
    plot = plot_curve(problem_name + "-Random Climb", "Iterations", "Fitness", range(fitness_curve.size), fitness_curve)
    time = end - start
    num_iters = fitness_curve.size
    return best_fitness, time, num_iters, plot


def solve_sim(problem, problem_name):
    start = timer()
    best_state, best_fitness, fitness_curve = simulated_annealing(problem, curve=True)
    end = timer()
    plot = plot_curve(problem_name + "-Simulated Annealing", "Iterations", "Fitness", range(fitness_curve.size), fitness_curve)
    time = end - start
    num_iters = fitness_curve.size
    return best_fitness, time, num_iters, plot


def solve_gen(problem, problem_name):
    start = timer()
    best_state, best_fitness, fitness_curve = genetic_alg(problem, curve=True)
    end = timer()
    plot = plot_curve(problem_name + "-Genetic", "Iterations", "Fitness", range(fitness_curve.shape[0]), np.max(fitness_curve, axis=1))
    time = end - start
    num_iters = fitness_curve.shape[0]
    return best_fitness, time, num_iters, plot


def solve_mimic(problem, problem_name):
    start = timer()
    best_state, best_fitness, fitness_curve = mimic(problem, curve=True)
    end = timer()
    plot = plot_curve(problem_name + "-MIMIC", "Iterations", "Fitness", range(fitness_curve.shape[0]), np.max(fitness_curve, axis=1))
    time = end - start
    num_iters = fitness_curve.shape[0]
    return best_fitness, time, num_iters, plot


problem_size = 100
# Optimization Example Problems
problems = {'OneMax': DiscreteOpt(problem_size, OneMax()),
            'FlipFlop': DiscreteOpt(problem_size, FlipFlop()),
            'FourPeaks': DiscreteOpt(problem_size, FourPeaks()),
            'Queens': DiscreteOpt(problem_size, Queens()),
            'Knapsack': DiscreteOpt(problem_size, Knapsack(list(np.random.randint(1, high=21, size=problem_size)),
                                                      list(np.random.randint(1, high=9, size=problem_size)))),
            'Custom': DiscreteOpt(problem_size, CustomFitness(custom_fn))}

# # One Max
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['OneMax'], 'OneMax')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['OneMax'], 'OneMax')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['OneMax'], 'OneMax')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['OneMax'], 'OneMax')

one_max_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
one_max_time = [hill_time, sim_time, gen_time, mimic_time]
one_max_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
one_max_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('OneMax\n')
for i in range(4):
    print('Hill: ', one_max_fitness[i], ' ', one_max_time[i], 's', ' ', one_max_iters[i], ' ')
    one_max_plot[i].savefig('one_max' + str(i) + '.png')

# Flip Flop
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['FlipFlop'], 'FlipFlop')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['FlipFlop'], 'FlipFlop')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['FlipFlop'], 'FlipFlop')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['FlipFlop'], 'FlipFlop')

FlipFlop_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
FlipFlop_time = [hill_time, sim_time, gen_time, mimic_time]
FlipFlop_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
FlipFlop_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('Flip Flop\n')
for i in range(4):
    print('Hill: ', FlipFlop_fitness[i], ' ', FlipFlop_time[i], 's', ' ', FlipFlop_iters[i], ' ')
    FlipFlop_plot[i].savefig('flip_flop' + str(i) + '.png')

# Four Peaks Problem (Simulated Annealing example)
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['FourPeaks'], 'FourPeaks')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['FourPeaks'], 'FourPeaks')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['FourPeaks'], 'FourPeaks')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['FourPeaks'], 'FourPeaks')

FourPeaks_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
FourPeaks_time = [hill_time, sim_time, gen_time, mimic_time]
FourPeaks_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
FourPeaks_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('Four Peaks\n')
for i in range(4):
    print('Hill: ', FourPeaks_fitness[i], ' ', FourPeaks_time[i], 's', ' ', FourPeaks_iters[i], ' ')
    FourPeaks_plot[i].savefig('four_peaks' + str(i) + '.png')

# 100-Queens Problem
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['Queens'], 'Queens')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['Queens'], 'Queens')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['Queens'], 'Queens')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['Queens'], 'Queens')

Queens_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
Queens_time = [hill_time, sim_time, gen_time, mimic_time]
Queens_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
Queens_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('Queen\n')
for i in range(4):
    print('Hill: ', Queens_fitness[i], ' ', Queens_time[i], 's', ' ', Queens_iters[i], ' ')
    Queens_plot[i].savefig('queen' + str(i) + '.png')

# Knapsack Problem
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['Knapsack'], 'Knapsack')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['Knapsack'], 'Knapsack')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['Knapsack'], 'Knapsack')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['Knapsack'], 'Knapsack')

Knapsack_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
Knapsack_time = [hill_time, sim_time, gen_time, mimic_time]
Knapsack_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
Knapsack_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('Knapsack\n')
for i in range(4):
    print('Hill: ', Knapsack_fitness[i], ' ', Knapsack_time[i], 's', ' ', Knapsack_iters[i], ' ')
    Knapsack_plot[i].savefig('knapsack' + str(i) + '.png')

# Adjusted OneMax Problem
hill_fitness, hill_time, hill_iters, hill_plot = solve_hill(problems['Custom'], 'Custom')
sim_fitness, sim_time, sim_iters, sim_plot = solve_sim(problems['Custom'], 'Custom')
gen_fitness, gen_time, gen_iters, gen_plot = solve_gen(problems['Custom'], 'Custom')
mimic_fitness, mimic_time, mimic_iters, mimic_plot = solve_mimic(problems['Custom'], 'Custom')

Custom_fitness = [hill_fitness, sim_fitness, gen_fitness, mimic_fitness]
Custom_time = [hill_time, sim_time, gen_time, mimic_time]
Custom_iters = [hill_iters, sim_iters, gen_iters, mimic_iters]
Custom_plot = [hill_plot, sim_plot, gen_plot, mimic_plot]

print('Custom\n')
for i in range(4):
    print('Hill: ', Custom_fitness[i], ' ', Custom_time[i], 's', ' ', Custom_iters[i], ' ')
    Custom_plot[i].savefig('custom' + str(i) + '.png')
