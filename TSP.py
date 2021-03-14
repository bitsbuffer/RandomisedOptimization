import mlrose_hiive as mlrose
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(1234)


def run_rhc(problem, initial_state):
    iterations = []
    attempts = []
    fitness = []

    def callback(**kwargs):
        if kwargs.get('attempt') is not None:
            iterations.append(kwargs['iteration'])
            attempts.append(kwargs['attempt'])
            fitness.append(kwargs['fitness'])
        if kwargs['fitness'] == 0:
            return False
        return True

    # Random hill climbling
    max_attempt = 40
    print(f"Running for max_attemp {max_attempt}")
    best_state, best_fitness, _ = mlrose.algorithms.random_hill_climb(problem,
                                                                      max_attempts=max_attempt,
                                                                      max_iters=1000,
                                                                      restarts=0,
                                                                      init_state=initial_state,
                                                                      state_fitness_callback=callback,
                                                                      callback_user_info=[],
                                                                      curve=False,
                                                                      random_state=1234)
    print(f"Best state -> {best_state}")
    print(f"Best fitness -> {best_fitness}")
    print("*****************************************************************************")
    return best_fitness, iterations, fitness


def run_ga(problem):
    # Genetic algorithm with max attempts
    iterations = []
    attempts = []
    fitness = []

    def callback(**kwargs):
        if kwargs.get('attempt') is not None:
            iterations.append(kwargs['iteration'])
            attempts.append(kwargs['attempt'])
            fitness.append(kwargs['fitness'])
        if kwargs['fitness'] == 0:
            return False
        return True

    max_attempts = 10
    mutation_prob = 0.1
    population_size = 30
    best_state, best_fitness, _ = mlrose.genetic_alg(problem,
                                                     pop_size=population_size,
                                                     mutation_prob=mutation_prob,
                                                     max_attempts=max_attempts,
                                                     max_iters=1000,
                                                     curve=False,
                                                     callback_user_info=[],
                                                     state_fitness_callback=callback,
                                                     random_state=1234)
    print(f"Best state -> {best_state}")
    print(f"Best fitness -> {best_fitness}")
    return best_fitness, iterations, fitness


def run_sa(problem, initial_state):
    iterations = []
    attempts = []
    fitness = []

    def callback(**kwargs):
        if kwargs.get('attempt') is not None:
            iterations.append(kwargs['iteration'])
            attempts.append(kwargs['attempt'])
            fitness.append(kwargs['fitness'])
        if kwargs['fitness'] == 0:
            return False
        return True
    schedule = mlrose.ExpDecay()
    max_attempt = 20
    best_state, best_fitness, _ = mlrose.simulated_annealing(problem,
                                                             schedule=schedule,
                                                             max_attempts=max_attempt,
                                                             max_iters=1000,
                                                             init_state=initial_state,
                                                             callback_user_info=[],
                                                             state_fitness_callback=callback,
                                                             random_state=1234)
    print(f"Best state -> {best_state}")
    print(f"Best fitness -> {best_fitness}")
    return best_fitness, iterations, fitness


def run_mimic(problem):
    # MIMIC with max attempts
    iterations = []
    attempts = []
    fitness = []

    def callback(**kwargs):
        if kwargs.get('attempt') is not None:
            iterations.append(kwargs['iteration'])
            attempts.append(kwargs['attempt'])
            fitness.append(kwargs['fitness'])
        if kwargs['fitness'] == 0:
            return False
        return True

    best_state, best_fitness, _ = mlrose.mimic(problem,
                                               pop_size=80,
                                               keep_pct=0.45,
                                               max_attempts=10,
                                               max_iters=1000,
                                               curve=False,
                                               state_fitness_callback=callback,
                                               callback_user_info=[],
                                               random_state=1234)
    print(f"Best state -> {best_state}")
    print(f"Best fitness -> {best_fitness}")
    return best_fitness, iterations, fitness


if __name__ == '__main__':
    # define problem
    dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426),
                 (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000),
                 (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426),
                 (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721),
                 (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056),
                 (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623),
                 (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
    fitness_coords = mlrose.TravellingSales(distances=dist_list)
    problem = mlrose.TSPOpt(length=8, fitness_fn=fitness_coords, maximize=False)
    initial_state = np.array([0, 1, 3, 5, 2, 4, 6, 7])
    # run rhc
    start_time = datetime.now()
    rhc_best_fit, rhc_iterations, rhc_fitness = run_rhc(problem, initial_state)
    rhc_end_time = (datetime.now() - start_time).total_seconds()
    # run sa
    start_time = datetime.now()
    sa_best_fit, sa_iterations, sa_fitness = run_sa(problem, initial_state)
    sa_end_time = (datetime.now() - start_time).total_seconds()
    # run ga
    start_time = datetime.now()
    ga_best_fit, ga_iterations, ga_fitness = run_ga(problem)
    ga_end_time = (datetime.now() - start_time).total_seconds()
    # run mimic
    start_time = datetime.now()
    mimic_best_fit, mimic_iterations, mimic_fitness = run_mimic(problem)
    mimic_end_time = (datetime.now() - start_time).total_seconds()
    # plot iterations vs fitness
    plt.figure()
    plt.plot(rhc_iterations, rhc_fitness, color="red", label="Randomised Hill Climbing")
    plt.plot(sa_iterations, sa_fitness, color="blue", label="Simulated Annealing")
    plt.plot(ga_iterations, ga_fitness, color="green", label="Genetic Algorithm")
    plt.plot(mimic_iterations, mimic_fitness, color="purple", label="MIMIC")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Comparison of algorithms on fitness wrt iterations")
    plt.legend()
    plt.savefig("./output/tsp/iterations-vs-fitness.png")
    plt.show()
    # time for executiom
    plt.figure()
    algorithms = ["RHC", "SA", "GA", "MIMIC"]
    time_execution = [rhc_end_time, sa_end_time, ga_end_time, mimic_end_time]
    plt.bar(algorithms, time_execution)
    plt.xlabel("Algorithms")
    plt.ylabel("Time for Execution in seconds")
    plt.title("Algorithms vs Time for Execution")
    plt.savefig("./output/tsp/algorithm-vs-execution-time.png")
    plt.show()
    # algorithms vs fit
    plt.figure()
    best_fit = [rhc_best_fit, sa_best_fit, ga_best_fit, mimic_best_fit]
    plt.bar(algorithms, best_fit)
    plt.xlabel("Algorithms")
    plt.ylabel("Best Fitness Score")
    plt.title("Algorithms vs Best Fitness Score")
    plt.savefig("./output/tsp/algorithm-vs-fittness-score.png")
    plt.show()
