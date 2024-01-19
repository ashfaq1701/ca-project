import numpy as np

from gol.gol import create_gol_instance_from_board
from gol.helpers import parse_pattern
from visualize import export_current_state


def add_pattern_to_board(board, pattern, row, col):
    board[row:row + pattern.shape[0], col:col + pattern.shape[1]] = pattern
    return board


def prepare_pattern_1(height, width, steps):
    glider_pattern = parse_pattern("../patterns/glider.txt")
    toad_pattern = parse_pattern("../patterns/toad.txt")
    beacon_pattern = parse_pattern("../patterns/beacon.txt")

    board = np.zeros((height, width), dtype=int)

    add_pattern_to_board(board, glider_pattern, 1, 2)
    add_pattern_to_board(board, glider_pattern, 14, 13)

    add_pattern_to_board(board, toad_pattern, 8, 2)
    add_pattern_to_board(board, toad_pattern, 5, 12)

    add_pattern_to_board(board, beacon_pattern, 14, 2)
    add_pattern_to_board(board, beacon_pattern, 1, 8)

    ca = create_gol_instance_from_board(board)
    before = ca.board.copy()
    ca.evolve_and_apply(steps)

    return before, ca


def prepare_pattern_2(height, width, steps):
    glider_pattern = parse_pattern("../patterns/glider.txt")
    beacon_pattern = parse_pattern("../patterns/beacon.txt")

    board = np.zeros((height, width), dtype=int)

    add_pattern_to_board(board, glider_pattern, 5, 2)
    add_pattern_to_board(board, glider_pattern, 14, 13)

    add_pattern_to_board(board, beacon_pattern, 8, 10)

    ca = create_gol_instance_from_board(board)
    before = ca.board.copy()
    ca.evolve_and_apply(steps)

    return before, ca


def get_and_save_pattern_1(height, width, steps):
    before, after = prepare_pattern_1(height, width, steps)

    export_current_state(
        before,
        "Pattern #1 Step 0",
        export_path="../../../data/gol-ga/raw/pattern_1_step_0.png"
    )

    export_current_state(
        after,
        f"Pattern #1 Step {steps}",
        export_path=f"../../../data/gol-ga/warmup/pattern_1_step_{steps}.png"
    )

    return after


def get_and_save_pattern_2(height, width, steps):
    before, after = prepare_pattern_2(height, width, steps)

    export_current_state(
        before,
        "Pattern #2 Step 0",
        export_path="../../../data/gol-ga/raw/pattern_2_step_0.png"
    )

    export_current_state(
        after,
        f"Pattern #2 Step {steps}",
        export_path=f"../../../data/gol-ga/warmup/pattern_2_step_{steps}.png"
    )

    return after
