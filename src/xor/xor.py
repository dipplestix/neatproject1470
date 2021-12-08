import torch


def get_fitness(model):
    inputs = torch.tensor([[1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
    outputs = model(inputs)
    outputs = torch.reshape(outputs, [1, 4])
    expected = torch.tensor([0., 1., 1., 0.])
    fitness = torch.sum(1 - torch.square(outputs - expected))

    return fitness
