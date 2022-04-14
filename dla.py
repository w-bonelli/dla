import csv
from uuid import uuid4
from random import random
from math import sqrt
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt


def start_position(ax: Optional[int] = None) -> np.ndarray:
    """
    Returns a random starting position [x, y] along one (randomly selected) of the grid's 4 axes.

    :param ax: The axis opposite which to bias movement
    :return: The position coordinate
    """

    cell = np.random.randint(0, 200)
    if ax is None: ax = np.random.randint(0, 3)
    if ax == 0: return np.array([0, cell])
    if ax == 2: return np.array([cell, 0])
    if ax == 1: return np.array([199, cell])
    if ax == 3: return np.array([cell, 199])
    raise ValueError(f"Invalid axis")


def distance_from_center(point, grid) -> float:
    """
    Calculates the Euclidean distance from the given point to the center of the grid.

    :param point: The point
    :param grid: The grid
    :return: The distance between the point and the center
    """

    h, w = grid.shape
    center = [h / 2, w / 2]
    return np.linalg.norm(np.array(point) - np.array(center))


def is_occupied(point, grid) -> bool:
    """
    Checks if the given point is occupied, i.e., is already part of the structure.

    :param point: The point's coordinates [x, y]
    :param grid: The grid
    :return: True if the point is on the structure, otherwise False
    """

    y, x = point
    return grid[y, x] == 1


def should_stick(p=0.5):
    """
    Determines whether to "stick" to the structure by flipping a coin biased with probability `p` of a positive result.

    :param p: The probability to stick to the structure.
    :return: Whether to stick to the structure.
    """

    return random() < p


id = str(uuid4())
g = np.zeros((200, 200))
g[100, 100] = 1
moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
walks = 1000

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im = plt.imshow(g, interpolation='bicubic', cmap='viridis')
# plt.ion()

for walk in range(0, walks):
    # choose a random start position on a random axis
    position = start_position()

    # walk until we stick to the structure somewhere, keeping track of steps
    steps = 0
    while True:
        steps += 1

        # choose a next step
        step = np.array(moves[np.random.choice(range(0, 4))])
        next_position = position + step

        # if the next location is beyond the grid boundaries, get another one
        if not (0 <= next_position[0] < 200 and 0 <= next_position[1] < 200): continue

        # if the new location is already on the structure, choose whether to stick to it
        if is_occupied(point=next_position, grid=g) and should_stick(p=0.8):
            g[position[0], position[1]] = 1

            # im.set_data(g)
            # plt.draw()
            # plt.pause(0.001)

            print(f"Walk {walk} attached after {steps} steps")
            break
        else:
            # otherwise keep walking
            position = next_position

# compute cluster size and radius of gyration
nonzero = np.nonzero(g)
points = [(a, b) for a, b in zip(nonzero[0], nonzero[1])]
radii = [distance_from_center(point=point, grid=g) for point in points]
S = len(points)
R = round(sqrt(sum([r ** 2 for r in radii]) / S), 3)
print(f"Finished {walks} walks, S = {S}, R = {R}")

# save S and R to CSV
with open(f"{id}.csv", 'w') as file:
    writer = csv.writer(file)
    writer.writerow([S, R])


# show the grid and save it to PNG
plt.text(100, 180, f"S = {S}, R = {R}", color='green')
plt.imshow(g, interpolation='bicubic', cmap='viridis')
plt.show()
fig.savefig(f"{id}.png")
# plt.waitforbuttonpress()
# plt.ioff()
