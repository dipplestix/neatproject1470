from typing import List
import random
import numpy as np


class GeneList:
    def __init__(self, genes: List):
        self.genes = genes
        self.nodes = set.union(set(g.n_in for g in genes), set(g.n_out for g in genes))
        self.inos = set([g.ino for g in genes])
        self.ino_dic = {g.ino: g for g in genes}
        self.directedConnects = set((g.n_in, g.n_out) for g in genes)
        self.active_connects = {(g.n_in, g.n_out): g.w for g in genes if g.active}


class Genome:
    def __init__(self, gene_list: GeneList, fitness: float, generation: int = 0):
        self.gene_list = gene_list
        self.fitness = fitness
        self.generation = generation

    def __gt__(self, other):
        return self.fitness >= other.fitness
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness


class Gene:
    def __init__(self, n_in: int, n_out: int, w: float, ino: int, active: bool):
        self.n_in = n_in
        self.n_out = n_out
        self.w = w
        self.ino = ino
        self.active = active

    def __str__(self):
        if self.active:
            active_stat = 'active'
        else:
            active_stat = 'disabled'
        return f'Connection between {self.n_in} to {self.n_out} has weight {self.w} and innovation number {self.ino}' \
               f'and is {active_stat}.'


class Species:
    def __init__(self, genome: List, gen):
        self.genomes = [genome]
        self.rep = genome
        self.champion = None
        self.gen = gen
        self.gen_improved = gen

    def find_champion(self):
        best = -np.inf
        for gen in self.genomes:
            if gen.fitness > best:
                best = gen.fitness
                self.champion = gen

    def find_rep(self):
        self.rep = random.choice(self.genomes)

    def next_generation(self):
        self.genomes = []
        self.best = -np.inf
        self.champion = None

    def add(self, genome):
        if self.champion is not None:
            if genome.fitness > self.champion.fitness:
                self.champion = genome
                self.gen_improved = genome.generation
        self.genomes.append(genome)
