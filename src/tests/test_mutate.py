from neat.neat_structures import Genome, Gene, GeneList
from neat.neat_functions import mutate_weights, new_link, new_node
from typing import List
from random import seed

connection0_2 = Gene(
    n_in=0,
    n_out=2,
    w=.5,
    ino=0,
    active=True
)

connection1_2 = Gene(
    n_in=1,
    n_out=2,
    w=.5,
    ino=1,
    active=True
)

connection2_3 = Gene(
    n_in=2,
    n_out=3,
    w=.5,
    ino=2,
    active=True
)

genes = [connection0_2, connection1_2, connection2_3]

genome = GeneList(genes)


#
#      (3)           
#       |    
#      (2)
#      / \ 
#     /   \
#   (1)   (0)


def test_mutateWeights():
    seed(10)
    new_genes = mutate_weights(genome)
    weights = sum(gene.w for gene in new_genes.genes)
    assert weights != 1.5


#     (3)
#    / |    
#   | (2)
#   | /  \ 
#   |/    \
#  (1)    (0)
def test_new_link1():
    seed(10)
    new_genes, ino, ino_dic = new_link(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2}, inputs=[0, 1], outputs=[3])
    assert 3 in new_genes.inos


def test_new_link2():
    new_genes, ino, ino_dic = new_link(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2}, inputs=[0, 1], outputs=[3])
    assert len(new_genes.genes) == 4


def test_new_link3():
    new_genes, ino, ino_dic = new_link(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2}, inputs=[0, 1], outputs=[3])
    assert new_genes.genes[-1].w == -0.7320549459017323


def test_new_link4():
    new_genes, ino, ino_dic = new_link(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2}, inputs=[0, 1], outputs=[3])
    assert (2, 3) in new_genes.directed_connects


#     (3)
#    / |  
#  (4) | 
#   | (2)
#   | /  \ 
#   |/    \
#  (1)    (0)
def test_new_node1():
    seed(10)
    new_genes, ino, ino_dic = new_node(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2})
    assert new_genes.inos == {0, 1, 2, 3, 4}


def test_new_node2():
    new_genes, ino, ino_dic = new_node(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2})
    assert len(new_genes.genes) == 5


def test_new_node3():
    new_genes, ino, ino_dic = new_node(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2})
    assert 4 in new_genes.nodes


def test_new_node4():
    new_genes, ino, ino_dic = new_node(genome, 3, {(0, 2): 0, (1, 2): 1, (2, 3): 2})
    assert (0, 4) and (4, 2) in new_genes.directed_connects
