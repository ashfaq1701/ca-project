import numpy as np


def parse_pattern(filepath, height, width):
    with open(filepath, "r") as f:
        full_text = f.read()

    pattern_arr = [[0 if ch == '.' else 1 for ch in line] for line in full_text.split("\n")]
    padded_pattern_arr = [pattern_row + [0] * max(width - len(pattern_row), 0) for pattern_row in pattern_arr]
    residue_array = np.zeros((height - len(padded_pattern_arr), width), dtype=int)
    return np.vstack([padded_pattern_arr, residue_array])
