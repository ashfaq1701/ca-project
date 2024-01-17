import numpy as np

import config
from gol.ga.ga_gol import GAGol
from wolfram.ga.ga_wolfram import GAWolfram
import matplotlib.pyplot as plt


def get_max_fitness_over_time(
        ca_name,
        patterns,
        population_size,
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
        fitness_over_time = None

        for repeat in range(repeats):
            print(f"\n\n#### Evolving Pattern {pattern_idx + 1}, Repeat {repeat} ####\n\n")

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

            current_fitness_over_time = np.array(ga.get_best_fitness_over_generations())
            if fitness_over_time is None:
                fitness_over_time = current_fitness_over_time
            else:
                fitness_over_time = np.mean(
                    [fitness_over_time, current_fitness_over_time],
                    axis=0
                )

        parsed_title = title.format(pattern_idx=pattern_idx + 1)
        parsed_export_path = export_path.format(pattern_idx=pattern_idx + 1)

        plot_fitness_over_time(fitness_over_time, parsed_title, parsed_export_path)


def plot_fitness_over_time(fitness_over_time, title, export_path):
    plt.figure()
    plt.plot(fitness_over_time, color='b')
    plt.axhline(y=np.nanmean(fitness_over_time), linestyle='--', color='r')
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    plt.title(title)
    plt.savefig(export_path)
    plt.show()
