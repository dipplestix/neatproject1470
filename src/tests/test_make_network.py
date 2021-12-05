from neat.neat_structures import Genome, Gene, GeneList
from neat.make_phenotype import make_network
import torch


gen1 = Gene(0, 2, 100, 0, True)
gen2 = Gene(1, 2, 30, 0, True)
genome_1 = GeneList([gen1])
genome_2 = GeneList([gen1, gen2])


def test_make_network_1():
    m = make_network(genome_1)
    assert m(torch.tensor([0.])) == torch.sigmoid(torch.tensor(0.))


def test_make_network_2():
    m = make_network(genome_1)
    assert m(torch.tensor([1.])) == torch.sigmoid(torch.tensor(100.))


def test_make_network_3():
    m = make_network(genome_2)
    assert m(torch.tensor([0.])) == torch.sigmoid(torch.tensor(0.))


def test_make_network_4():
    m = make_network(genome_2)
    assert m(torch.tensor([1.])) == torch.sigmoid(torch.tensor(130.))


def test_make_network_5():
    m = make_network(genome_2)
    assert m(torch.tensor([.1])) == torch.sigmoid(torch.tensor(13.))
