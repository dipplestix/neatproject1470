from xor.xor import get_fitness
import torch


def model1(inputs):
    return torch.tensor([0., 1., 1., 0.])


def model2(inputs):
    return torch.tensor([1., 3., -1., -1.])


def model3(inputs):
    return torch.tensor([0., 100., 1., 0.])


def test_xor1():
    fitness = get_fitness(model1)
    assert fitness == 0


def test_xor2():
    fitness = get_fitness(model2)
    assert fitness == -10


def test_xor3():
    fitness = get_fitness(model3)
    assert fitness == -9801
