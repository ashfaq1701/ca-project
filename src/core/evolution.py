from gol.ga.ga_gol import GAGol
from visualize import visualize_current_state, export_current_state
from wolfram.ca import WolframCA
from wolfram.ga.ga_wolfram import GAWolfram


def evolve_and_visualize_patterns(
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
        raw_export_directory,
        raw_plot_title,
        evolved_export_directory,
        evolved_plot_title,
        repeats=1,
        callback_function=None,
        callback_interval=None
):
    if ca_name == 'game_of_life':
        class_obj = GAGol
    else:
        class_obj = GAWolfram

    for repeat in range(repeats):
        for idx, ca in enumerate(patterns):
            print(f"\n\n#### Evolving Pattern {idx + 1} ####\n\n")

            ga = class_obj(
                target=ca.board,
                population_size=population_size,
                n_generations=n_generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                retention_rate=retention_rate,
                random_selection_rate=random_selection_rate,
                steps=steps,
                fitness_function_name=fitness_function_name,
                callback_function=callback_function,
                callback_interval=callback_interval
            )

            ga.evolution()

            top_performing_cas = ga.get_top_n_fittest_individuals(3)

            for best_fit_idx, ca_item in enumerate(top_performing_cas):
                evolved_board = ca_item['individual'].evolve(5)
                rule_number = ca_item['individual'].rule_number if isinstance(ca_item['individual'], WolframCA) else None

                visualize_current_state(evolved_board, ca_item['fitness'], idx + 1)

                raw_directory = format_with_placeholders(
                    raw_export_directory,
                    ca_item['fitness'],
                    steps,
                    idx + 1,
                    best_fit_idx + 1,
                    rule_number,
                    repeat
                )

                raw_title = format_with_placeholders(
                    raw_plot_title,
                    ca_item['fitness'],
                    steps,
                    idx + 1,
                    best_fit_idx + 1,
                    rule_number,
                    repeat
                )

                evolved_directory = format_with_placeholders(
                    evolved_export_directory,
                    ca_item['fitness'],
                    steps,
                    idx + 1,
                    best_fit_idx + 1,
                    rule_number,
                    repeat
                )

                evolved_title = format_with_placeholders(
                    evolved_plot_title,
                    ca_item['fitness'],
                    steps,
                    idx + 1,
                    best_fit_idx + 1,
                    rule_number,
                    repeat
                )

                export_current_state(
                    ca_item['individual'],
                    raw_title,
                    export_path=raw_directory
                )

                export_current_state(
                    evolved_board,
                    evolved_title,
                    export_path=evolved_directory
                )


def format_with_placeholders(str_prop,
                             fitness,
                             ca_steps,
                             pattern_idx,
                             best_fit_idx,
                             rule_number,
                             repeat_number):
    return str_prop.format(
        fitness=round(fitness, 2),
        ca_steps=ca_steps,
        pattern_idx=pattern_idx,
        best_fit_idx=best_fit_idx,
        rule_number=rule_number,
        repeat_number=repeat_number
    )