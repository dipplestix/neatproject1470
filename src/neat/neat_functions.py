from neat.neat_structures import Genome, Gene
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List


def initialization(n_inputs: int, n_outputs: int, get_fitness: Callable, pop_size: int = 150):
    ino = 0
    population_genomes = [[] for _ in range(pop_size)]

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
                population_genomes[k].append(new_gene)
            ino += 1

    population = []
    for gen in population_genomes:
        fitness = get_fitness(gen)
        population.append(Genome(gen, fitness, generation=0))

    return population


def breed(g1: Genome, g2: Genome, get_fitness: Callable, generation: int) -> Genome:
    """
    :param g1: Genome of the first parent
    :param g2: Genome of the second parent
    :param get_fitness: Genome of the second parent
    :param generation: Current generation
    :return: Genome of the child
    """
    if g1>g2:
        better_parent = g1
        other_parent = g2
    else:
        better_parent = g2
        other_parent = g1
    genome_dic = deepcopy(better_parent.ino_dic)
    for ino in genome_dic:
        if ino in other_parent.inos:
            if not better_parent.ino_dic[ino].active or not other_parent.ino_dic[ino].active:
                if random()<.75:
                    genome_dic[ino].active = False
                else:
                    genome_dic[ino].active = True
            genome_dic[ino].w = choice([better_parent.ino_dic[ino].w, other_parent.ino_dic[ino].w])

    genome = list(genome_dic.values())

    fitness = get_fitness(genome)

    child = Genome(genome, fitness, generation)

    return child


def delta(genome1: Genome, genome2: Genome, c1: float = 1.0, c2: float = 1.0, c3: float = .4):
    """
    :param c1: Excess coefficient
    :param c2: Disjoint coefficient
    :param c3: Matching coefficient
    :param genome1: First genome
    :param genome2: Second genome
    :return: Compatibility distance between genomes
    """
    genes1 = genome1.genes
    genes2 = genome2.genes
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
    while i<len(genes1) or j<len(genes2):
        if i>=len(genes1):
            excess += (len(genes2) - j)
            break
        elif j>=len(genes2):
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
        elif gene1.ino<gene2.ino:
            disjoint += 1
            i += 1
        else:
            disjoint += 1
            j += 1

    # sanity check
    assert matching != 0

    delta = c1*(excess/n) + c2*(disjoint/n) + c3*(total_diff/matching)

    return delta


###__Weight Mutation__###


def mutateWeights(genes: List):
    """
	Function to randomly mutates the genome to alter the connection weights
	:param genes: List of genes
	"""
    for connection in genes:
        if random()<0.8:
            if random()<0.9:
                randomPerturbation = uniform(-0.05, 0.05)
                connection.w += randomPerturbation
            else:
                newWeight = uniform(-1, 1)
                connection.w = newWeight


###__Structural Mutations__###


def mutateConnection(g: Genome):
    """
	Function to randomly mutates the genome to add a new connection
	:param genes: Genome to mutate
	"""

    # ADD CONNECTION
    if random()<0.75:
        newConnection = False
        while not newConnection:
            toBeConnected = sample(g.nodes, 2)  # Get random new nodes to connect
            node1, node2 = toBeConnected[0], toBeConnected[1]
            if (node1, node2) in g.directedConnects:  # If existing connection, start over 
                continue
            newConnection = True

        randomWeight = uniform(-1, 1)  # Get new Weight

        ####TODO: figure out innovation number
        ino = max(g.inos)
        ino += 1
        newConnection = Gene(node1, node2, randomWeight, ino, active=True)
        g.genes.append(newConnection)
        g.inos.add(ino)
        g.ino_dic.update({ino: newConnection})
        g.directedConnects.add((node1, node2))


def mutateNode(g: Genome):
    """
	Function to randomly mutates the genome to add a new node
	:param genes: Genome to mutate	
	"""

    # ADD NODE
    if random()<0.75:
        connection = sample(g.genes, 1)[0]  # Get connection in which to insert node 
        connection.active = False  # Disable old connection
        oldWeight = connection.w
        newWeight = 1
        node1, node2 = connection.n_in, connection.n_out
        g.directedConnects.remove((node1, node2))  # Remove directed connection

        newNode = len(g.nodes)  # Get number for new node
        g.nodes.add(newNode)

        ino = max(g.inos)
        ino += 1
        newConnection1 = Gene(node1, newNode, newWeight, ino, active=True)  # Connect node1 and new node
        g.genes.append(newConnection1)
        g.inos.add(ino)
        g.ino_dic.update({ino: newConnection1})
        g.directedConnects.add((node1, newNode))

        ino += 1
        newConnection2 = Gene(newNode, node2, oldWeight, ino, active=True)  # connect new node and node2
        g.genes.append(newConnection2)
        g.inos.add(ino)
        g.ino_dic.update({ino: newConnection2})
        g.directedConnects.add((newNode, node2))
