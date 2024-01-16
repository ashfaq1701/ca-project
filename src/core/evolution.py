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
        fitness_function_name
):
    if ca_name == 'game_of_life':
        class_obj = GAGol
        directory_name = "gol-ga"
    else:
        class_obj = GAWolfram
        directory_name = "wolfram-ga"

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
            fitness_function_name=fitness_function_name
        )

        ga.evolution()

        top_performing_cas = ga.get_top_n_fittest_individuals(3)

        for best_fit_idx, ca_item in enumerate(top_performing_cas):
            evolved_board = ca_item['individual'].evolve(5)
            rule_number = ca_item['individual'].rule_number if isinstance(ca_item['individual'], WolframCA) else None

            visualize_current_state(evolved_board, ca_item['fitness'], idx + 1)

            if fitness_function_name == 'mae':
                directory = f"../../../data/{directory_name}/evolution/mae"
            else:
                directory = f"../../../data/{directory_name}/evolution/structural_similarity"

            label = f"Pattern {idx + 1}, Best Fit {best_fit_idx},"

            if rule_number is None:
                initial_title = f"{label} Initial States"
                evolved_title = f"{label} After {steps} Steps Fitness: {round(ca_item['fitness'], 2)}"
            else:
                initial_title = f"{label} Rule {rule_number}, Initial States"
                evolved_title = f"{label} Rule {rule_number}, After {steps} Steps Fitness: {round(ca_item['fitness'], 2)}"

            export_current_state(
                ca_item['individual'],
                initial_title,
                export_path=f"{directory}/pattern_{idx + 1}_best_fit_{best_fit_idx + 1}_initial_states.png"
            )

            export_current_state(
                evolved_board,
                evolved_title,
                export_path=f"{directory}/pattern_{idx + 1}_best_fit_{best_fit_idx + 1}_after_steps_{steps}.png"
            )
