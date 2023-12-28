from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from gol.gol import GameOfLife


def generator(ca: GameOfLife, generations):
    for generation in range(generations + 1):
        if generation > 0:
            ca.evolve_step()

        yield ca.board


def update(frame, img):
    img.set_array(frame)
    return img,


def evolve_and_visualize(ca: GameOfLife, generations):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Game of Life Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(ca.board, cmap="Blues")

    animation = FuncAnimation(
        fig,
        update,
        fargs=(img,),
        frames=generator(ca, generations),
        interval=200,
        save_count=generations + 1
    )
    return animation


def evolve_and_visualize_at_end(ca: GameOfLife, generations):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Game of Life Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])

    ca.evolve(generations)
    ax.imshow(ca.board, cmap="Blues")
    plt.show()
