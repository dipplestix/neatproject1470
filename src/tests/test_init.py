from neat.neat_structures import Genome, Gene
from neat.neat_functions import initialization
from typing import List


def get_fitness(genome: List):
    return 12


def test_init():
    pop = initialization(5, 3, get_fitness, pop_size=10)
    assert len(pop) == 10


def test_init2():
    pop = initialization(5, 3, get_fitness, pop_size=10)
    assert len(pop[0].genes) == 15
