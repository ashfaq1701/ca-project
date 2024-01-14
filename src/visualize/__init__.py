from core.ca import CA
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def generator(ca: CA, steps):
    current_board = None

    for step in range(steps + 1):
        if step > 0:
            current_board = ca.evolve_step(step - 1, current_board)
            ca.apply(current_board)

        yield ca.board


def update(frame, img):
    img.set_array(frame)
    return img,


def evolve_and_visualize(ca: CA, steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(ca.board, cmap="Blues")

    animation = FuncAnimation(
        fig,
        update,
        fargs=(img,),
        frames=generator(ca, steps),
        interval=200,
        save_count=steps + 1
    )
    return animation


def evolve_and_visualize_at_end(ca: CA, steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cellular Automaton")
    ax.set_xticks([])
    ax.set_yticks([])

    ca.evolve_and_apply(steps)
    ax.imshow(ca.board, cmap="Blues")
    plt.show()


def visualize_current_state(ca, fitness=None):
    title = "Cellular Automaton"

    if fitness:
        title = f"{title}, Fitness = {fitness}"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if isinstance(ca, CA):
        board = ca.board
    else:
        board = ca

    ax.imshow(board, cmap="Blues")
    plt.show()
