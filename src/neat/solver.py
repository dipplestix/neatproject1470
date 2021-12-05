from neat.neat_structures import Genome, Gene, Species
from neat.neat_functions import *
from typing import Callable
import numpy as np
import random
import time


def solve(n_in: int, n_out: int, get_fitness: Callable, last_layer_func: Callable, n_pop: int =150, max_t: int =300):

    # Initialize the population and some parameters
    generation = 0
    pop, ino = initialization(n_in, n_out, get_fitness, n_pop)
    start = time.time()
    delta_thresh = 3
    species_list = np.zeros(n_pop)

    # Initiate the first species and add everything to it
    first = pop[0]

    species = [Species(first)]
    for ind in pop[1:]:
        species[0].add(ind)

    # Run for however long we tell it to
    while time.time() - start < max_t:
        # Increment the generation and reset the population
        generation += 1
        new_pop = []

        # Find the champion and representative for each species and reset them
        for s in species:
            s.find_champion()
            s.find_rep()
            new_pop.append(s.champion)
            s.next_generation()

        while len(new_pop) < n_pop:
            ind = random.choice(pop)
            if random.random() < .8:



        # Loop through and check for speciation
        for ind in pop[1:]:
            in_species = False
            for s in species:
                delt = delta(ind.gene_list, s.rep.gene_list)
                if delt<delta_thresh:
                    species.add(ind)
                    in_species = True
                    break
            if not in_species:
                species.append(Species(ind))





