from neat.neat_structures import Genome, Gene, GeneList
from neat.make_phenotype import make_network
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List


def initialization(n_inputs: int, n_outputs: int, get_fitness: Callable, pop_size: int = 150):
    ino = 0
    population_gene_lists = [[] for _ in range(pop_size)]

    # Make each genome gene-by-gene with random weights
    for i in range(n_inputs):
        for j in range(n_outputs):
            for k in range(pop_size):
                new_gene = Gene(n_in=i,
                                n_out=j,
                                w=uniform(-1, 1),
                                ino=ino,
                                active=True
                                )
                population_gene_lists[k].append(new_gene)
            ino += 1

    population = []
    for gen in population_gene_lists:
        gene_list = GeneList(gen)
        model = make_network(gene_list)
        fitness = get_fitness(model)
        population.append(Genome(gene_list, fitness, generation=0))

    return population


def breed(g1: Genome, g2: Genome) -> List:
    """
    :param g1: Genome of the first parent
    :param g2: Genome of the second parent
    :return: List of genes of the child
    """
    if g1 > g2:
        better_parent = g1
        other_parent = g2
    else:
        better_parent = g2
        other_parent = g1
    genome_dic = deepcopy(better_parent.gene_list.ino_dic)
    for ino in genome_dic:
        if ino in other_parent.gene_list.inos:
            if not better_parent.gene_list.ino_dic[ino].active or not other_parent.gene_list.ino_dic[ino].active:
                if random() < .75:
                    genome_dic[ino].active = False
                else:
                    genome_dic[ino].active = True
            genome_dic[ino].w = choice([better_parent.gene_list.ino_dic[ino].w, other_parent.gene_list.ino_dic[ino].w])

    gene_list = GeneList(list(genome_dic.values()))

    return gene_list


def delta(gene_list1: GeneList, gene_list2: GeneList, c1: float = 1.0, c2: float = 1.0, c3: float = .4):
    """
    :param c1: Excess coefficient
    :param c2: Disjoint coefficient
    :param c3: Matching coefficient
    :param gene_list1: First genome
    :param gene_list2: Second genome
    :return: Compatibility distance between genomes
    """
    genes1 = gene_list1.genes
    genes2 = gene_list2.genes
    n = max(len(genes1), len(genes2))

    # counts
    excess = 0
    disjoint = 0
    matching = 0

    # total difference between matching genes
    total_diff = 0

    # find excess and disjoint count
    i = 0
    j = 0
    while i < len(genes1) or j < len(genes2):
        if i >= len(genes1):
            excess += (len(genes2) - j)
            break
        elif j >= len(genes2):
            excess += (len(genes1) - i)
            break

        gene1 = genes1[i]
        gene2 = genes2[j]

        if gene1.ino == gene2.ino:
            # calculate diff
            diff = abs(gene1.w - gene2.w)
            total_diff += diff
            matching += 1
            i += 1
            j += 1
        elif gene1.ino < gene2.ino:
            disjoint += 1
            i += 1
        else:
            disjoint += 1
            j += 1

    # sanity check
    assert matching != 0, 'Should have at least 1 matching node'
    
    delta_val = c1*(excess/n) + c2*(disjoint/n) + c3*(total_diff/matching)

    return delta_val


def mutate_weights(genes: List):
    """
    Function to randomly mutates the genome to alter the connection weights
    :param genes: List of genes
    """
    for connection in genes:
        if random() < 0.8:
            if random() < 0.9:
                random_perturbation = uniform(-0.05, 0.05)
                connection.w += random_perturbation
            else:
                new_weight = uniform(-1, 1)
                connection.w = new_weight


def mutate_connection(g: Genome):
    """
    Function to randomly mutates the genome to add a new connection
    :param g: Genome to mutate
    """

    # ADD CONNECTION
    if random() < 0.75:
        new_connection = False
        while not new_connection:
            to_be_connected = sample(g.nodes, 2)  # Get random new nodes to connect
            node1, node2 = to_be_connected[0], to_be_connected[1]
            if (node1, node2) in g.directedConnects:  # If existing connection, start over 
                continue
            new_connection = True

        random_weight = uniform(-1, 1)  # Get new Weight

        # TODO: figure out innovation number
        ino = max(g.inos)
        ino += 1
        new_connection = Gene(node1, node2, random_weight, ino, active=True)
        g.genes.append(new_connection)
        g.inos.add(ino)
        g.ino_dic.update({ino: new_connection})
        g.directedConnects.add((node1, node2))


def mutate_node(g: Genome):
    """
    Function to randomly mutates the genome to add a new node
    :param g: Genome to mutate	
    """

    # ADD NODE
    if random() < 0.75:
        connection = sample(g.genes, 1)[0]  # Get connection in which to insert node 
        connection.active = False  # Disable old connection
        old_weight = connection.w
        new_weight = 1
        node1, node2 = connection.n_in, connection.n_out
        g.directedConnects.remove((node1, node2))  # Remove directed connection

        new_node = len(g.nodes)  # Get number for new node
        g.nodes.add(new_node)

        ino = max(g.inos)
        ino += 1
        new_connection1 = Gene(node1, new_node, new_weight, ino, active=True)  # Connect node1 and new node
        g.genes.append(new_connection1)
        g.inos.add(ino)
        g.ino_dic.update({ino: new_connection1})
        g.directedConnects.add((node1, new_node))

        ino += 1
        new_connection2 = Gene(new_node, node2, old_weight, ino, active=True)  # connect new node and node2
        g.genes.append(new_connection2)
        g.inos.add(ino)
        g.ino_dic.update({ino: new_connection2})
        g.directedConnects.add((new_node, node2))
