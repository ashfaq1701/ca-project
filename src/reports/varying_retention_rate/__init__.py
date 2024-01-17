import numpy as np

import config
from gol.ga.ga_gol import GAGol
from wolfram.ga.ga_wolfram import GAWolfram
import matplotlib.pyplot as plt


def get_fitness_by_varying_retention_rates(
        ca_name,
        patterns,
        population_size,
        n_generations,
        crossover_rate,
        mutation_rate,
        retention_rates,
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
        fitness_with_retention_rates = None

        for repeat in range(repeats):
            print(f"\n\n#### Evolving Pattern {pattern_idx + 1}, Repeat {repeat} ####\n\n")
            current_fitness_with_retention_rates = []

            for retention_rate in retention_rates:
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

                current_fitness_with_retention_rates.append(ga.get_best_fitness())

            current_fitness_with_retention_rates = np.array(current_fitness_with_retention_rates)
            if fitness_with_retention_rates is None:
                fitness_with_retention_rates = current_fitness_with_retention_rates
            else:
                fitness_with_retention_rates = np.mean(
                    [fitness_with_retention_rates, current_fitness_with_retention_rates],
                    axis=0
                )

        parsed_title = title.format(pattern_idx=pattern_idx + 1)
        parsed_export_path = export_path.format(pattern_idx=pattern_idx + 1)

        plot_fitness_vs_retention_rates(fitness_with_retention_rates, list(retention_rates), parsed_title, parsed_export_path)


def plot_fitness_vs_retention_rates(fitness_with_retention_rates, retention_rates, title, export_path):
    plt.figure()
    plt.plot(retention_rates, fitness_with_retention_rates, color='b')
    plt.axhline(y=np.nanmean(fitness_with_retention_rates), linestyle='--', color='r')
    plt.xlabel('Retention Rate')
    plt.ylabel('Max Fitness')
    plt.title(title)
    plt.savefig(export_path)
    plt.show()