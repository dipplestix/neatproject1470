from neat.neat_structures import Genome, Gene, Species
from neat.neat_functions import *
from typing import Callable
from copy import deepcopy
import numpy as np
from random import random
import time


def solve(n_in: int, n_out: int, get_fitness: Callable, last_layer_func: Callable, n_pop: int = 150, max_t: int = 10):
    # Initialize the population and some parameters
    inputs = list(range(n_in))
    outputs = list(range(n_in, n_in + n_out))
    generation = 0
    pop, ino = initialization(n_in, n_out, get_fitness, n_pop, last_layer_func)
    start = time.time()
    delta_thresh = 3

    best_fit = -np.inf
    best = None

    for ind in pop:
        if ind.fitness>best_fit:
            best_fit = ind.fitness
            best = ind

    # Initiate the first species and add everything to it
    first = pop[0]

    species = [Species(first, generation)]
    for ind in pop[1:]:
        species[0].add(ind)
    species_dic = {i: species[0] for i in range(n_pop)}

    # Run for however long we tell it to
    while time.time() - start<max_t:
        ino_dic = {}
        new_species_dic = {}
        generation += 1
        new_pop = []
        new_gene_lists = []
        print(f'Generation {generation}: {len(species)}')

        for i, s in enumerate(species):
            if len(s.genomes)>0:
                s.find_champion()
                s.find_rep()
                new_pop.append(s.champion)
                new_species_dic[i] = s

        # good_species = {}
        # one_good = False
        # for i, s in enumerate(species):
        #     if len(s.genomes) > 0:
        #         if s.gen_improved > generation - 15:
        #             good_species[i] = True
        #             one_good = True
        #         else:
        #             good_species[i] = False
        #     else:
        #         good_species[i] = False
        # if one_good == False:
        #     champions = []
        #     for s in species:
        #         if len(s.genomes) > 0:
        #             try:
        #                 champions.append(s.champion.fitness)
        #             except:
        #                 print(s.genomes)
        #     best_species = champions.index(max(champions))
        #     good_species[best_species] = True

        while len(new_gene_lists)<n_pop:
            # Go through the mutation and breeding process
            index = choice(list(range(n_pop)))
            # keep = False
            # while not keep:
            #     if good_species[species.index(species_dic[index])] == True:
            #         keep = True
            #     else:
            #         index = choice(list(range(n_pop)))
            ind = pop[index]
            fit = ind.fitness
            new_gene_list = ind.gene_list
            if random()<.8:
                new_gene_list = mutate_weights(ind.gene_list)
            if random()<.05:
                new_gene_list, ino, ino_dic = new_link(new_gene_list, ino, ino_dic, inputs, outputs)
            if random()<.05:
                new_gene_list, ino, ino_dic = new_node(new_gene_list, ino, ino_dic)
            if random()<.75:
                if random()<0.001:
                    other = choice(pop)
                else:
                    other = choice(species_dic[index].genomes)
                other_genes = other.gene_list
                other_fit = other.fitness
                fit = get_fitness(make_network(new_gene_list, inputs, outputs))
                new_gene_list = breed(new_gene_list, other_genes, fit, other_fit)

            new_gene_lists.append(new_gene_list)

        new_species = species
        for s in new_species:
            s.next_generation()

        for i, gene_list in enumerate(new_gene_lists):
            # Assign to a species
            in_species = False
            for s in new_species:
                delt = delta(new_gene_list, s.rep.gene_list)
                if delt<delta_thresh:
                    s.add(ind)
                    in_species = True
                    new_species_dic[i] = s
                    break
            if not in_species:
                # print('I made a new species')
                new_species.append(Species(ind, ind.generation))
                new_species_dic[i] = new_species[-1]

        for i, gene_list in enumerate(new_gene_lists):
            # Build model and find fitness
            model = make_network(new_gene_list, inputs, outputs, last_layer_func)
            new_fit = get_fitness(model)
            new_ind = Genome(new_gene_list, new_fit/len(new_species_dic[i].genomes), generation)

            if new_fit>best_fit:
                print(new_fit)
                best_fit = new_fit
                best = new_ind

            new_pop.append(new_ind)

        # Get ready for next generation
        pop = new_pop
        species = new_species
        species_dic = new_species_dic

    return best, generation
