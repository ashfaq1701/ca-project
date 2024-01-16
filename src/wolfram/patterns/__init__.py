import numpy as np

from visualize import export_current_state
from wolfram.ca import create_wolfram_ca_instance_from_pattern_and_rule_number


def prepare_pattern_1(width, steps):
    init_states = np.zeros(width, dtype=int)
    init_states[width // 2] = 1

    ca = create_wolfram_ca_instance_from_pattern_and_rule_number(init_states, 30)
    before = ca.board.copy()
    ca.evolve_and_apply(steps)

    return before, ca


def prepare_pattern_2(width, steps):
    init_states = np.zeros(width, dtype=int)
    init_states[4] = 1
    init_states[width // 3 + 2] = 1
    init_states[(width // 6) * 5 + 2] = 1

    ca = create_wolfram_ca_instance_from_pattern_and_rule_number(init_states, 90)
    before = ca.board.copy()
    ca.evolve_and_apply(steps)

    return before, ca


def get_and_save_pattern_1(width, steps):
    before, after = prepare_pattern_1(width, steps)

    export_current_state(
        before,
        "Pattern #1 Step 0",
        export_path="../../../data/wolfram-ga/raw/pattern_1_step_0.png"
    )

    export_current_state(
        after,
        f"Pattern #1 Step {steps}",
        export_path=f"../../../data/wolfram-ga/warmup/pattern_1_step_{steps}.png"
    )

    return after


def get_and_save_pattern_2(width, steps):
    before, after = prepare_pattern_2(width, steps)

    export_current_state(
        before,
        "Pattern #2 Step 0",
        export_path="../../../data/wolfram-ga/raw/pattern_2_step_0.png"
    )

    export_current_state(
        after,
        f"Pattern #2 Step {steps}",
        export_path=f"../../../data/wolfram-ga/warmup/pattern_2_step_{steps}.png"
    )

    return after
