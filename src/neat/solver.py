from neat.neat_structures import Genome, Gene, Species
from neat.neat_functions import *
from typing import Callable
from copy import deepcopy
import numpy as np
import random
import time


def solve(n_in: int, n_out: int, get_fitness: Callable, last_layer_func: Callable, n_pop: int =150, max_t: int =300):

    # Initialize the population and some parameters
    inputs = list(range(n_in))
    outputs = list(range(n_in, n_out))
    generation = 0
    pop, ino = initialization(n_in, n_out, get_fitness, n_pop)
    start = time.time()
    delta_thresh = 3
    species_dic = {i: 0 for i in pop}

    best_fit = -np.inf
    best = None

    for ind in pop:
        if ind.fitness > best_fit:
            best_fit = ind.fitness
            best = ind

    # Initiate the first species and add everything to it
    first = pop[0]

    species = [Species(first)]
    for ind in pop[1:]:
        species[0].add(ind)

    # Run for however long we tell it to
    while time.time() - start < max_t:
        ino_dic = {}
        new_species = deepcopy(species)
        new_species_dic = {}
        # Increment the generation and reset the population
        generation += 1
        new_pop = []
        new_gene_lists = []

        for s in new_species:
            s.find_champion()
            s.find_rep()
            s.next_generation()
            new_pop.append(s.champion)

        while len(new_pop) < n_pop:
            # Go through the mutation and breeding process
            ind = random.choice(pop)
            fit = ind.fitness
            new_gene_list = ind.gene_list
            if random.random() < .8:
                new_gene_list = mutate_weights(ind.gene_list)
            if random.random() < .05:
                new_gene_list, ino, ino_dic = new_link(new_gene_list, ino, ino_dic, inputs, outputs)
            if random.random() < .05:
                new_gene_list, ino, ino_dic = new_node(new_gene_list, ino, ino_dic)
            if random.random() < .75:
                if random.random() < 0.001:
                    other = random.sample(pop)
                else:
                    other = random.sample(species_dic[ind].genomes)
                other_genes = other.gene_list
                other_fit = other.fitness
                new_gene_list = breed(new_gene_list, other_genes, fit, other_fit)

            # Assign to a species
            in_species = False
            for s in new_species:
                delt = delta(new_gene_list, s.rep.gene_list)
                if delt < delta_thresh:
                    s.add(ind)
                    in_species = True
                    new_species_dic[ind] = s
                    break
            if not in_species:
                new_species.append(Species(ind))
                new_species_dic[ind] = new_species[-1]

            new_gene_lists.append(new_gene_list)
        for gene_list in new_gene_lists:
            # Build model and find fitness
            model = make_network(new_gene_list)
            new_fit = get_fitness(model)
            new_ind = Genome(new_gene_list, new_fit/len(new_species_dic[gene_list].genomes))

            if new_fit > best_fit:
                best_fit = new_fit
                best = new_ind

            new_pop.append(new_ind)


        # Get ready for next generation
        pop = new_pop
        species = new_species
        species_dic = new_species_dic

    return best

