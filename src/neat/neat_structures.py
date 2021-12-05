from typing import List


class Genome:
    def __init__(self, genes: List, fitness: float, generation: int = 0):
        self.genes = genes 
        self.nodes = set.union(set(g.n_in for g in genes), set(g.n_out for g in genes))
        self.fitness = fitness
        self.generation = generation
        self.inos = set([g.ino for g in genes])
        self.ino_dic = {g.ino: g for g in genes}
        self.directedConnects = set((g.n_in, g.n_out) for g in genes)
        self.active_connects = {(g.n_in, g.n_out): g.w for g in genes if g.active}
        
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
    def __init__(self, genomes: List, champion: Genome):
        self.genomes = genomes
        self.champion = champion
