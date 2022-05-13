import csv
import argparse
from uuid import uuid4
from random import random
from math import sqrt
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def get_start_position(side: int, ax: Optional[int] = None) -> np.ndarray:
    """
    Returns a random starting position [x, y] along one (randomly selected) of the grid's 4 axes.

    :param side: The length of a side of the square grid
    :param ax: The axis opposite which to bias movement
    :return: The position coordinate
    """

    cell = np.random.randint(0, side)
    if ax is None: ax = np.random.randint(0, 3)
    if ax == 0: return np.array([0, cell])
    if ax == 2: return np.array([cell, 0])
    if ax == 1: return np.array([side - 1, cell])
    if ax == 3: return np.array([cell, side - 1])
    raise ValueError(f"Invalid axis")


def get_next_step(side: int, point) -> np.ndarray:
    """
    Randomly selects a step within the boundaries of the grid.

    :param side: The length of a side of the square grid
    :param point: The current location of the walker
    :return: The next step
    """

    # moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    moves = []
    if point[0] != 0: moves.append([-1, 0])
    if point[1] != 0: moves.append([0, -1])
    if point[0] != side - 1: moves.append([1, 0])
    if point[1] != side - 1: moves.append([0, 1])
    return np.array(moves[np.random.choice(range(0, len(moves)))])


def distance_from_center(side: int, point) -> float:
    """
    Calculates the Euclidean distance from the given point to the center of the grid.

    :param side: The length of a side of the square grid
    :param point: The current location of the walker
    :return: The distance between the point and the center
    """

    half = int(side / 2)
    center = [half, half]
    return np.linalg.norm(np.array(point) - np.array(center))


def is_occupied(g: np.ndarray, point) -> bool:
    """
    Checks if the given point is occupied, i.e., is already part of the DLA cluster.

    :param g: The grid
    :param point: The coordinates [x, y] of the point to check
    :return: True if the point is on the cluster, otherwise False
    """

    y, x = point
    return g[y, x] == 1


def should_stick(p: float = 0.5):
    """
    Determines whether to "stick" to the cluster by flipping a coin biased with probability `p` of a positive result.

    :param p: The probability to stick to the cluster.
    :return: Whether to stick to the cluster.
    """

    return random() < p


def dla(side: int, mass: int, prob: float):
    """
    Runs a DLA simulation trial.

    :param side: Length of a side of the square grid
    :param mass: How many walkers to attach to the cluster
    :param prob: Probability of a walker to stick to an adjacent cluster cell
    """

    # unique ID to distinguish this trial
    guid = str(uuid4())

    # initialize the grid
    grid = np.zeros((side, side))

    # set the cluster seed in the middle of the grid
    half = int(side / 2)
    grid[half, half] = 1

    # initialize the plot
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    plt.imshow(grid, interpolation='bicubic', cmap='viridis')

    for walk in range(0, mass - 1):
        # choose a random start position on a random axis
        position = get_start_position(side=side)

        # walk until we stick to the structure somewhere, keeping track of steps
        steps = 0
        while True:
            steps += 1

            # choose a next step
            step = get_next_step(side=side, point=position)
            next_position = position + step

            # if the next location is beyond the grid boundaries, get another one
            if not (0 <= next_position[0] < side and 0 <= next_position[1] < side):
                print("Stepped out of bounds! This should never happen. Probably a bug in step selection logic.")
                continue

            # if the new location is already on the structure, choose whether to stick to it
            if is_occupied(g=grid, point=next_position) and should_stick(p=prob):
                grid[position[0], position[1]] = 1
                print(f"Walk {walk} attached after {steps} steps")
                break
            else:
                # otherwise keep walking
                position = next_position

    # compute cluster mass (i.e. size, S) and mean radius of gyration (R)
    nonzero = np.nonzero(grid)
    points = [(a, b) for a, b in zip(nonzero[0], nonzero[1])]
    radii = [distance_from_center(side=side, point=point) for point in points]
    S = len(points)
    R = round(sqrt(sum([r ** 2 for r in radii]) / S), 3)
    print(f"Finished trial {guid} on {side}x{side} grid with {mass} walks, S = {S}, R = {R}")

    # save S and R to CSV
    with open(f"{guid}.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow([S, R])

    # save the grid to PNG
    plt.text(half, int(half / 5), f"S = {S}, R = {R}", color='green')
    plt.imshow(grid, interpolation='bicubic', cmap='viridis')
    plt.show()
    fig.savefig(f"{guid}.png")


if __name__ == '__main__':
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--side", type=int, default=100, help="Length of a side of the square grid")
    ap.add_argument("-m", "--mass", type=int, default=100, help="How many walkers to attach to the cluster")
    ap.add_argument("-p", "--prob", type=float, default=1.0, help="Probability of a walker to stick to an adjacent cluster cell")
    args = vars(ap.parse_args())

    # run the DLA simulation
    dla(side=int(args['side']),
        mass=int(args['mass']),
        prob=int(args['prob']))
