import config
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
    if ca_name == config.CA_NAME_GAME_OF_LIFE:
        class_obj = GAGol
    else:
        class_obj = GAWolfram

    for repeat in range(repeats):
        for idx, ca in enumerate(patterns):
            print(f"\n\n#### Evolving Pattern {idx + 1} ####\n\n")

            checkpoint_fn_with_params = None
            if callback_function is not None:
                checkpoint_fn_with_params = callback_function(idx)

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
                callback_function=checkpoint_fn_with_params,
                callback_interval=callback_interval
            )

            ga.evolution()

            top_performing_cas = ga.get_top_n_fittest_individuals(3)

            for best_fit_idx, ca_item in enumerate(top_performing_cas):
                evolved_board = ca_item['individual'].evolve(5)
                rule_number = ca_item['individual'].rule_number if isinstance(ca_item['individual'],
                                                                              WolframCA) else None

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


def checkpoint_save_function(target_dir_before, title_before, target_dir_after, title_after):
    def checkpoint_with_pattern(pattern_idx):
        def checkpoint(generation, fittest_ind, best_fitness, steps):
            print(f"Saving checkpoint for Pattern#{pattern_idx + 1} in {generation}")

            rule_number = fittest_ind.rule_number if isinstance(fittest_ind, WolframCA) else None

            target_dir_before_parsed = format_with_placeholders(target_dir_before,
                                                                best_fitness,
                                                                steps,
                                                                pattern_idx,
                                                                None,
                                                                rule_number,
                                                                None,
                                                                generation)

            title_before_parsed = format_with_placeholders(title_before,
                                                           best_fitness,
                                                           steps,
                                                           pattern_idx,
                                                           None,
                                                           rule_number,
                                                           None,
                                                           generation)

            target_dir_after_parsed = format_with_placeholders(target_dir_after,
                                                               best_fitness,
                                                               steps,
                                                               pattern_idx,
                                                               None,
                                                               rule_number,
                                                               None,
                                                               generation)

            title_after_parsed = format_with_placeholders(title_after,
                                                          best_fitness,
                                                          steps,
                                                          pattern_idx,
                                                          None,
                                                          rule_number,
                                                          None,
                                                          generation)

            export_current_state(
                fittest_ind,
                title_before_parsed,
                export_path=target_dir_before_parsed
            )

            evolved_board = fittest_ind.evolve(steps)

            export_current_state(
                evolved_board,
                title_after_parsed,
                export_path=target_dir_after_parsed
            )

        return checkpoint

    return checkpoint_with_pattern


def format_with_placeholders(str_prop,
                             fitness,
                             ca_steps,
                             pattern_idx,
                             best_fit_idx,
                             rule_number,
                             repeat_number,
                             generation=None):
    return str_prop.format(
        fitness=round(fitness, 2),
        ca_steps=ca_steps,
        pattern_idx=pattern_idx,
        best_fit_idx=best_fit_idx,
        rule_number=rule_number,
        repeat_number=repeat_number,
        generation=generation
    )
