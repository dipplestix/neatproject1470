from neat.neat_structures import Genome, Gene, GeneList
from neat.make_phenotype import make_network
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List


def initialization(n_inputs: int, n_outputs: int, get_fitness: Callable, pop_size: int = 150, last_layer=torch.sigmoid):
    ino = 0
    population_gene_lists = [[] for _ in range(pop_size)]

    # Make each genome gene-by-gene with random weights
    for i in range(n_inputs):
        for j in range(n_inputs, n_outputs + n_inputs):
            for k in range(pop_size):
                new_gene = Gene(n_in=i,
                                n_out=j,
                                w=uniform(-10, 10),
                                ino=ino,
                                active=True
                                )
                population_gene_lists[k].append(new_gene)
            ino += 1

    population = []
    for gen in population_gene_lists:
        gene_list = GeneList(gen)
        model = make_network(gene_list, list(range(n_inputs)), list(range(n_inputs, n_outputs + n_inputs)), last_layer)
        fitness = get_fitness(model)
        population.append(Genome(gene_list, fitness/pop_size, generation=0))

    return population, ino


def breed(g1: GeneList, g2: GeneList, f1: float, f2: float) -> List:
    """
    :param g1: Genome of the first parent
    :param g2: Genome of the second parent
    :param f1: Genome of the first parent
    :param f2: Genome of the second parent
    :return: List of genes of the child
    """
    if f1>f2:
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
    assert matching != 0, 'Should have at least 1 matching node'
    if n<20:
        n = 1
    delta_val = c1*(excess/n) + c2*(disjoint/n) + c3*(total_diff/matching)

    return delta_val


def mutate_weights(gene_list: GeneList):
    """
    Function to randomly mutates the genome to alter the connection weights
    :param gene_list: List of genes
    """
    new_genes = deepcopy(gene_list.genes)
    for connection in new_genes:
        if random()<0.9:
            random_perturbation = uniform(-1, 1)
            connection.w += random_perturbation
        else:
            new_weight = uniform(-10, 10)
            connection.w = new_weight

    return GeneList(new_genes)


def new_link(gene_list: GeneList, ino: int, ino_dic: dict, inputs: List, outputs: List):
    """
    Function to randomly mutates the genome to add a new connection
    :param gene_list: GeneList to mutate
    :param ino: innovation number
    :param ino_dic: innovation dictionary to check if it has already been made
    :param inputs: list of input nodes
    :param outputs: list of output nodes

    """

    new_connection = False
    try_count = 0
    while not new_connection and try_count<20:
        to_be_connected = sample(list(gene_list.nodes), 2)  # Get random new nodes to connect
        node1, node2 = to_be_connected[0], to_be_connected[1]
        if (node1,
            node2) in gene_list.directed_connects or node2 in inputs or node1 in outputs:  # If existing connection, start over
            try_count += 1
            continue
        # Check for loop
        nodes_to_consider = [n2 for (n1, n2) in gene_list.directed_connects if n1 == node1]
        nodes_to_consider.append(node2)
        while nodes_to_consider:
            if node1 in nodes_to_consider and try_count<20:
                try_count += 1
                continue
            node = nodes_to_consider.pop()
            extras = [n2 for (n1, n2) in gene_list.directed_connects if n1 == node]
            nodes_to_consider.extend(extras)

        new_connection = True

    if try_count<20:
        try:
            ino_num = ino_dic[(node1, node2)]
        except KeyError:
            ino_num = ino
            ino_dic[(node1, node2)] = ino
            ino += 1

        random_weight = uniform(-1, 1)  # Get new Weight

        # print(f'new connection: {(node1, node2)}')
        new_connection = Gene(node1, node2, random_weight, ino_num, active=True)
        new_genes = deepcopy(gene_list.genes)

        new_genes.append(new_connection)
    else:
        new_genes = gene_list.genes
    return GeneList(new_genes), ino, ino_dic


def new_node(gene_list: GeneList, ino: int, ino_dic: dict):
    """
    Function to randomly mutates the genome to add a new node
    :param gene_list: GeneList to mutate
    :param ino: innovation number
    :param ino_dic: innovation dictionary to check if it has already been made
    """
    new_genes = deepcopy(gene_list.genes)
    connection = choice(new_genes)  # Get connection in which to insert node
    connection.active = False  # Disable old connection
    old_weight = connection.w
    new_weight = 1
    node1, node2 = connection.n_in, connection.n_out

    new_node_num = len(gene_list.nodes)  # Get number for new node

    try:
        ino_num = ino_dic[(node1, new_node_num)]
    except KeyError:
        ino_num = ino
        ino_dic[(node1, new_node_num)] = ino
        ino_dic[(new_node_num, node2)] = ino + 1
        ino += 2

    new_connection1 = Gene(node1, new_node_num, new_weight, ino_num, active=True)  # Connect node1 and new node
    new_genes.append(new_connection1)

    new_connection2 = Gene(new_node_num, node2, old_weight, ino_num + 1, active=True)  # connect new node and node2
    new_genes.append(new_connection2)

    return GeneList(new_genes), ino, ino_dic
