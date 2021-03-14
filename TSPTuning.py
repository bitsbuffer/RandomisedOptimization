import mlrose_hiive as mlrose
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

np.random.seed(1234)


def plot(x, y1, y2, title, xlabel, y1_label, y2_label, fname, text1):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    plt.subplot(axes)
    plt.plot(x, y1)
    plt.xlabel(xlabel)
    plt.ylabel(y1_label)
    plt.text(0.98, 0.98, text1,
             horizontalalignment='right',
             verticalalignment='top',
             transform=axes.transAxes)
    # plt.subplot(axes[1])
    # plt.plot(x, y2)
    # plt.xlabel(xlabel)
    # plt.ylabel(y2_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def run_rhc(problem, initial_state, prob_name):
    # Random hill climblinga
    max_attempts = np.linspace(5, 50, 10)
    hc_best_fit = []
    hc_time_taken = []
    for max_attempt in max_attempts:
        start_time = datetime.now()
        print(f"Running for max_attemp {max_attempt}")
        best_state, best_fitness, _ = mlrose.algorithms.random_hill_climb(problem,
                                                                          max_attempts=max_attempt,
                                                                          max_iters=1000,
                                                                          restarts=0,
                                                                          init_state=initial_state,
                                                                          curve=False,
                                                                          random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        hc_best_fit.append(best_fitness)
        hc_time_taken.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for Random Hill climbing {hc_best_fit}")
    print("*****************************************************************************")
    # code for plots
    plot(max_attempts, hc_best_fit, hc_time_taken,
         "Randomised Hill Climbing with varying max attempts",
         "Max Attempts",
         "Fitness",
         "Time Taken",
         "./output/"+ prob_name + "/rhc_max_attempts.png",
         "Max Iteration=1000\nRestarts=0")

    # tune restarts
    restarts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    hc_best_restart_fit = []
    hc_time_restart_taken = []
    for restart in restarts:
        start_time = datetime.now()
        print(f"Running for restart {restart}")
        best_state, best_fitness, _ = mlrose.algorithms.random_hill_climb(problem,
                                                                          max_attempts=40,
                                                                          max_iters=1000,
                                                                          restarts=restart,
                                                                          init_state=initial_state,
                                                                          curve=False,
                                                                          random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        hc_best_restart_fit.append(best_fitness)
        hc_time_restart_taken.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for Random Hill climbing {hc_best_fit}")
    print("*****************************************************************************")
    # code for plots
    plot(restarts, hc_best_restart_fit, hc_time_taken,
         "Randomised Hill Climbing with varying Restarts",
         "Restarts",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/rhc_restart.png",
         "Max Iteration=1000\nMax Attempts=40")

    return hc_best_fit, hc_time_taken


def run_sa(problem, initial_state, prob_name):
    # simulated annealing with exponential decay varying max attempts
    max_attempts = np.linspace(10, 100, 10)
    sa_best_fit_exp_decay = []
    sa_time_taken_exp_decay = []
    schedule = mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
    for max_attempt in max_attempts:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.simulated_annealing(problem,
                                                                 schedule=schedule,
                                                                 max_attempts=max_attempt,
                                                                 max_iters=1000,
                                                                 init_state=initial_state,
                                                                 random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        sa_best_fit_exp_decay.append(best_fitness)
        sa_time_taken_exp_decay.append((datetime.now() - start_time).total_seconds())
    print("*****************************************************************************")
    print(f"Best fitness for simulated annealing with exponential decay {sa_best_fit_exp_decay}")

    # code for plots
    plot(max_attempts, sa_best_fit_exp_decay, sa_time_taken_exp_decay,
         "SA with varying max attempts with exponential decay",
         "Max Attempts",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/sa_ex_decay_max_attempts.png",
         "Max Iterations=2000 \nInit temp=1.0 \nMinTemp=0.001 \nExponential Constant=0.005")

    # simulated annealing with geometric decay varying max attempts
    start_time = datetime.now()
    sa_best_fit_geom_decay = []
    sa_time_taken_geom_decay = []
    schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
    for max_attempt in max_attempts:
        best_state, best_fitness, _ = mlrose.simulated_annealing(problem,
                                                                 schedule=schedule,
                                                                 max_attempts=max_attempt,
                                                                 max_iters=2000,
                                                                 init_state=initial_state,
                                                                 random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        sa_best_fit_geom_decay.append(best_fitness)
        sa_time_taken_geom_decay.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for simulated annealing with geometric decay {sa_best_fit_geom_decay}")
    print("*****************************************************************************")

    # code for plots
    plot(max_attempts, sa_best_fit_geom_decay, sa_time_taken_geom_decay,
         "SA with varying max attempts with Geometric decay",
         "Max Attempts",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/sa_geom_decay_max_attempts.png",
         "Max Iterations=2000 \nInit temp=1.0 \nMinTemp=0.001 \nDecay=0.99")

    return sa_best_fit_exp_decay, sa_time_taken_exp_decay, sa_best_fit_geom_decay, sa_time_taken_geom_decay


def run_ga(problem, prob_name):
    # Genetic algorithm with max attempts
    max_attempts = np.linspace(5, 50, 10)
    ga_best_fit_max_attempt = []
    ga_time_taken_max_attempt = []
    for max_attempt in max_attempts:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.genetic_alg(problem,
                                                         pop_size=100,
                                                         mutation_prob=0.1,
                                                         max_attempts=max_attempt,
                                                         max_iters=1000,
                                                         curve=False,
                                                         random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        ga_best_fit_max_attempt.append(best_fitness)
        ga_time_taken_max_attempt.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for genetic algorithm with varying max attempts {ga_best_fit_max_attempt}")
    print("*****************************************************************************")

    # code for plots
    plot(max_attempts, ga_best_fit_max_attempt, ga_time_taken_max_attempt,
         "Genetic algorithm with varying max attempts",
         "Max Attempts",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/ga_max_attempts.png",
         "Population Size=100 \nMutation Prob=0.1 \nMax Iterations=1000")

    # Genetic algorithm with pop size
    pop_sizes = np.linspace(10, 100, 10)
    ga_best_fit_pop_size = []
    ga_time_taken_pop_size = []
    for pop_size in pop_sizes:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.genetic_alg(problem,
                                                         pop_size=pop_size,
                                                         mutation_prob=0.1,
                                                         max_attempts=10,
                                                         max_iters=1000,
                                                         curve=False,
                                                         random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        ga_best_fit_pop_size.append(best_fitness)
        ga_time_taken_pop_size.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for genetic algorithm with varying population size {ga_best_fit_pop_size}")
    print("*****************************************************************************")
    # plot code
    plot(pop_sizes, ga_best_fit_pop_size, ga_time_taken_pop_size,
         "Genetic algorithm with varying Population size",
         "Population Size",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/ga_prop_size.png",
         "Max Attempts=10 \nMutation Prob=0.1 \nMax Iterations=1000")

    # Genetic algorithm with mutuation probability
    mutations_probs = np.linspace(0.1, 1, 10)
    ga_best_fit_mut_prob = []
    ga_time_taken_mut_prob = []
    for mutation_prob in mutations_probs:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.genetic_alg(problem,
                                                         pop_size=100,
                                                         mutation_prob=mutation_prob,
                                                         max_attempts=10,
                                                         max_iters=1000,
                                                         curve=False,
                                                         random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        ga_best_fit_mut_prob.append(best_fitness)
        ga_time_taken_mut_prob.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for genetic algorithm varying mutation probability is {ga_best_fit_mut_prob}")
    print("*****************************************************************************")
    # plot code
    plot(mutations_probs, ga_best_fit_mut_prob, ga_time_taken_mut_prob,
         "Genetic algorithm with varying mutation probability",
         "Mutation Probability",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/ga_mp.png",
         "Population Size=100 \nMax Attempts=10 \nMax Iterations=1000")
    return ga_best_fit_max_attempt, ga_time_taken_max_attempt, ga_best_fit_pop_size, ga_time_taken_pop_size, ga_best_fit_mut_prob, ga_time_taken_mut_prob


def run_mimic(problem, prob_name):
    # MIMIC with max attempts
    max_attempts = np.linspace(5, 50, 10)
    mimic_best_fit_max_attempt = []
    mimic_time_taken_max_attempt = []
    for max_attempt in max_attempts:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.mimic(problem,
                                                   pop_size=100,
                                                   keep_pct=0.1,
                                                   max_attempts=max_attempt,
                                                   max_iters=1000,
                                                   curve=False,
                                                   random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        mimic_best_fit_max_attempt.append(best_fitness)
        mimic_time_taken_max_attempt.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for mimic with varying max attempts {mimic_best_fit_max_attempt}")
    print("*****************************************************************************")
    # plot code
    plot(max_attempts, mimic_best_fit_max_attempt, mimic_time_taken_max_attempt,
         "MIMIC with varying Max Attempts",
         "Max Attempts",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/mimic_max_attempts.png",
         "Population Size=100 \nKeep PCT=0.1 \nMax Iterations=1000")

    # MIMIC with varying population size
    pop_sizes = np.linspace(10, 100, 10)
    mimic_best_fit_pop_size = []
    mimic_time_taken_pop_size = []
    for pop_size in pop_sizes:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.mimic(problem,
                                                   pop_size=pop_size,
                                                   keep_pct=0.1,
                                                   max_attempts=10,
                                                   max_iters=1000,
                                                   curve=False,
                                                   random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        mimic_best_fit_pop_size.append(best_fitness)
        mimic_time_taken_pop_size.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for mimic with varying population size {mimic_best_fit_pop_size}")
    print("*****************************************************************************")
    # plot code
    plot(pop_sizes, mimic_best_fit_pop_size, mimic_time_taken_pop_size,
         "MIMIC with varying Population Size",
         "Population Size",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/mimic_prop_size.png",
         "Max Attempts=10 \nKeep PCT=0.1 \nMax Iterations=1000")

    # Genetic algorithm with mutation probability
    keep_pcts = np.linspace(0.1, 0.5, 10)
    mimic_best_fit_keep_pct = []
    mimic_time_taken_keep_pct = []
    for keep_pct in keep_pcts:
        start_time = datetime.now()
        best_state, best_fitness, _ = mlrose.mimic(problem,
                                                   pop_size=80,
                                                   keep_pct=keep_pct,
                                                   max_attempts=50,
                                                   max_iters=1000,
                                                   curve=False,
                                                   random_state=1234)
        print(f"Best state -> {best_state}")
        print(f"Best fitness -> {best_fitness}")
        mimic_best_fit_keep_pct.append(best_fitness)
        mimic_time_taken_keep_pct.append((datetime.now() - start_time).total_seconds())
    print(f"Best fitness for MIMIC varying proportion of samples to keep at each iteration is {mimic_best_fit_keep_pct}")
    print("*****************************************************************************")
    # plot code
    plot(keep_pcts, mimic_best_fit_keep_pct, mimic_time_taken_keep_pct,
         "MIMIC with varying proportion of samples to keep at each iteration",
         "Keep Percentages",
         "Fitness",
         "Time Taken",
         "./output/" + prob_name + "/mimic_keep_samples.png",
         "Pop Size=80 \nMax Attempts=50 \nMax Iterations=1000")
    return mimic_best_fit_max_attempt, mimic_time_taken_max_attempt, mimic_best_fit_pop_size, mimic_time_taken_pop_size, mimic_best_fit_keep_pct, mimic_time_taken_keep_pct


def get_tsp():
    # dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
    #          (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4),
    #          (5, 2, 5), (5, 4, 7), (3, 5, 6), (5, 0, 4), (1, 5, 3)]
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
    return problem, initial_state


def run_all_algos(problem, initial_state, prob_name):
    hc_best_fit, hc_time_taken = run_rhc(problem, initial_state, prob_name)
    sa_best_fit_exp_decay, sa_time_taken_exp_decay, sa_best_fit_geom_decay, \
    sa_time_taken_geom_decay = run_sa(problem, initial_state, prob_name)
    ga_best_fit_max_attempt, ga_time_taken_max_attempt, ga_best_fit_pop_size, \
    ga_time_taken_pop_size, ga_best_fit_mut_prob, ga_time_taken_mut_prob = run_ga(problem, prob_name)
    mimic_best_fit_max_attempt, mimic_time_taken_max_attempt, mimic_best_fit_pop_size, \
    mimic_time_taken_pop_size, mimic_best_fit_keep_pct, mimic_time_taken_keep_pct = run_mimic(problem, prob_name)


if __name__ == '__main__':
    problem, initial_state = get_tsp()
    run_all_algos(problem, initial_state, "tsp")