from neat.neat_structures import Genome, Gene
from neat.neat_functions import *
from typing import Callable
import time


def solve(n_in: int, n_out: int, get_fitness: Callable, last_layer_func: Callable, pop: int =150, max_t: int =300):

    # Initialize the population and some parameters
    generation = 0
    pop, ino = initialization(n_in, n_out, get_fitness, pop)
    start = time.time()
    species = []
    first = pop.pop()


    # Run for 5 minutes (could
    while time.time() - start < max_t:



