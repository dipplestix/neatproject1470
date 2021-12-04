from neat.neat_structures import Genome, Gene
from neat.neat_functions import mutate_weights, mutate_connection, mutate_node
from typing import List
from random import seed

connection0_2 = Gene(
	n_in = 0, 
	n_out = 2, 
	w = .5, 
	ino=0,
	active = True
)

connection1_2 = Gene(
	n_in = 1, 
	n_out = 2, 
	w = .5, 
	ino=1,
	active = True
)

connection2_3 = Gene(
	n_in = 2, 
	n_out = 3, 
	w = .5, 
	ino=2,
	active = True
)

genes = [connection0_2, connection1_2, connection2_3]

genome = Genome(genes, fitness=0)

#
#      (3)           
#       |    
#      (2)
#      / \ 
#     /   \
#   (1)   (0)


def test_mutateWeights():
	seed(10)
	mutate_weights(genome.genes)
	weights = sum(gene.w for gene in genome.genes)
	assert weights != 1.5

#     (3)
#    / |    
#   | (2)
#   | /  \ 
#   |/    \
#  (1)    (0)
def test_mutateConnection1():
	seed(10)
	mutate_connection(genome)
	assert 3 in genome.inos
	
def test_mutateConnection2():
	assert len(genome.genes) == 4

def test_mutateConnection3():
	assert genome.genes[-1].w == 0.15618260226894076

def test_mutateConnection4():
	assert (3,1) in genome.directedConnects

#     (3)
#    / |  
#  (4) | 
#   | (2)
#   | /  \ 
#   |/    \
#  (1)    (0)
def test_mutateNode1():
	seed(10)
	mutate_node(genome)
	assert genome.inos == {0,1,2,3,4,5}

def test_mutateNode2():
	assert len(genome.genes) == 6

def test_mutateNode3():
	assert 4 in genome.nodes

def test_mutateNode4():
	assert (3,4) and (4,1) in genome.directedConnects