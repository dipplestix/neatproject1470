from xor.xor import get_fitness
import torch


def model1(inputs):
    return torch.tensor([0., 1., 1., 0.])


def model2(inputs):
    return torch.tensor([1., .5, .5, 1.])


def model3(inputs):
    return torch.tensor([0., 0., 1., 0.])


def test_xor1():
    fitness = get_fitness(model1)
    assert fitness == 4


def test_xor2():
    fitness = get_fitness(model2)
    assert fitness == 1.5


def test_xor3():
    fitness = get_fitness(model3)
    assert fitness == 3
