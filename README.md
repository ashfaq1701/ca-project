# Cellular Automata Pattern Evolution Using Genetic Algorithms

In this project I implemented two cellular automata implementations.

1. Game of Life
2. Wolfram Cellular Automata

Both were developed using common interfaces keeping their interoperability in mind. [GameOfLife](src/gol/gol.py) and [WolframCA](src/wolfram/ca.py) are the classes where these CA rules have been implemented.

Then using Genetic Algorithm I tried to evolve the GA with a predefined target. The predefined target is a CA state after running the CA for a specific number of states.

Two fitness functions are used while running the GA, MAE and [structural similarity](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html).

The GA is implemented using hierarchical classes. The base class is [GA](src/ga/__init__.py). Then both [GAGol](src/gol/ga/ga_gol.py) and [GAWolfram](src/wolfram/ga/ga_wolfram.py) contains the specific GA implementations.

Each of these GA classes are used to run their own set of experiments. The Jupyter notebooks associated with [game of life](src/gol/notebooks) and [Wolfram cellular automata](src/wolfram/notebooks) are in their respective hyperlinked directories.

Two nice animations of evolving [game of life](data/animation_gol.mp4) and [wolfram ca](data/animation_wolfram.mp4) can be found in their hyperlinked locations. They are produce using these notebooks corresponding to [game of life](src/gol/notebooks/playground.ipynb) and [wolfram ca](src/wolfram/notebooks/playground.ipynb).

The results of all the experiments are exported to this [data](data/) directory.

