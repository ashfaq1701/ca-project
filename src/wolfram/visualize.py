from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from wolfram.ca import WolframCA


def generator(ca):
    for step in range(ca.steps + 1):
        if step > 0:
            ca.evolve_step(step - 1)

        yield ca.board


def update(frame, img):
    img.set_array(frame)
    return img,


def evolve_and_visualize(ca: WolframCA):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Wolfram Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(ca.board, cmap="Blues")

    animation = FuncAnimation(
        fig,
        update,
        fargs=(img,),
        frames=generator(ca),
        interval=200,
        save_count=ca.steps + 1
    )
    return animation


def evolve_and_visualize_at_end(ca: WolframCA):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Wolfram Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])

    ca.evolve()
    ax.imshow(ca.board, cmap="Blues")
    plt.show()
