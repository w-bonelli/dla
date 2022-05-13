# DLA Simulation

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Installation](#installation)
- [Usage](#usage)
- [Optimizations](#optimizations)
- [Outputs](#outputs)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

A basic diffusion-limited aggregation simulation on a square grid.

## Installation

The most portable way to run this simulation is Docker. A public image is available on Docker Hub at [`wbonelli/dla`](https://hub.docker.com/r/wbonelli/dla).

To use Python instead of Docker, make a fresh environment with your tool of choice (e.g. `venv`, Anaconda), then install the packages in `requirements.txt` (e.g., `pip install -r requirements.txt`).

## Usage

At its simplest, the script can be invoked with:

```shell
python dla.py
```

By default a 100x100 grid will be used, with 100 particles (walkers), and sticking probability unity. Each of these parameters can be provided explicitly as well:

- `--side (-s)`: The length of a side of the square grid
- `--mass (-m)`: The number of walkers to attach to the cluster
- `--prob (-p)`: The probability of a walker to stick to an adjacent cluster cell

For instance, to use a 200x200 grid with 1000 walkers and a 50/50 chance of sticking:

```shell
python dla.py -s 200 -m 1000 -p 0.5
```

## Optimizations

None, save for boundary-aware step selection, i.e., not wasting cycles bumping into walls.

## Outputs

A GUID is generated each time the script is invoked. This GUID is written with status messages to `stdout` and forms the stem of the names of the output files produced by each invocation.

When a cluster is completed and a trial ends, the cluster's mass (M) and mean radius of gyration (R) are calculated, printed, and persisted to file:

- `{guid}.png`: a plot of the fully grown cluster (with M and R overlaid)
- `{guid}.csv`: a 1-line, 2-value CSV file whose first value is M, second is R

These quantities are helpful in estimating fractal dimension.
