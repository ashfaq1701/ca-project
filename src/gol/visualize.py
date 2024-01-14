import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from gol.gol import GameOfLife


def generator(ca: GameOfLife, steps):
    for step in range(steps + 1):
        if step > 0:
            ca.evolve_step(step)

        yield ca.board


def update(frame, img):
    img.set_array(frame)
    return img,


def evolve_and_visualize(ca: GameOfLife, steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Game of Life Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(ca.board, cmap="Blues")

    animation = FuncAnimation(
        fig,
        update,
        fargs=(img,),
        frames=generator(ca, steps),
        interval=300,
        save_count=steps + 1
    )
    return animation


def evolve_and_visualize_at_end(ca: GameOfLife, steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Game of Life Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])

    ca.evolve(steps)
    ax.imshow(ca.board, cmap="Blues")
    plt.show()


def parse_pattern(filepath, height, width):
    with open(filepath, "r") as f:
        full_text = f.read()

    pattern_arr = [[0 if ch == '.' else 1 for ch in line] for line in full_text.split("\n")]
    padded_pattern_arr = [pattern_row + [0] * max(width - len(pattern_row), 0) for pattern_row in pattern_arr]
    residue_array = np.zeros((height - len(padded_pattern_arr), width), dtype=int)
    return np.vstack([padded_pattern_arr, residue_array])
