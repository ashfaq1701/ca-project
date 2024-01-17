import numpy as np

import config
from gol.ga.ga_gol import GAGol
from wolfram.ga.ga_wolfram import GAWolfram
import matplotlib.pyplot as plt


def get_fitness_by_varying_population_size(
        ca_name,
        patterns,
        population_sizes,
        n_generations,
        crossover_rate,
        mutation_rate,
        retention_rate,
        random_selection_rate,
        steps,
        fitness_function_name,
        repeats,
        title,
        export_path
):
    if ca_name == config.CA_NAME_GAME_OF_LIFE:
        class_obj = GAGol
    else:
        class_obj = GAWolfram

    for pattern_idx, ca in enumerate(patterns):
        fitness_with_population_sizes = None

        for repeat in range(repeats):
            print(f"\n\n#### Evolving Pattern {pattern_idx + 1}, Repeat {repeat} ####\n\n")
            current_fitness_with_population_sizes = []

            for population_size in population_sizes:
                ga = class_obj(
                    target=ca.board,
                    population_size=population_size,
                    n_generations=n_generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    retention_rate=retention_rate,
                    random_selection_rate=random_selection_rate,
                    steps=steps,
                    fitness_function_name=fitness_function_name
                )

                ga.evolution()

                current_fitness_with_population_sizes.append(ga.get_best_fitness())

            current_fitness_with_population_sizes = np.array(current_fitness_with_population_sizes)
            if fitness_with_population_sizes is None:
                fitness_with_population_sizes = current_fitness_with_population_sizes
            else:
                fitness_with_population_sizes = np.mean(
                    [fitness_with_population_sizes, current_fitness_with_population_sizes],
                    axis=0
                )

        parsed_title = title.format(pattern_idx=pattern_idx + 1)
        parsed_export_path = export_path.format(pattern_idx=pattern_idx + 1)

        plot_fitness_vs_population_sizes(fitness_with_population_sizes, list(population_sizes), parsed_title, parsed_export_path)


def plot_fitness_vs_population_sizes(fitness_with_population_sizes, population_sizes, title, export_path):
    plt.figure()
    plt.plot(population_sizes, fitness_with_population_sizes, color='b')
    plt.axhline(y=np.nanmean(fitness_with_population_sizes), linestyle='--', color='r')
    plt.xlabel('Population')
    plt.ylabel('Max Fitness')
    plt.title(title)
    plt.savefig(export_path)
    plt.show()