import torch


def get_fitness(model):
    inputs = torch.tensor([[1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
    outputs = model(inputs)
    expected = torch.tensor([0., 1., 1., 0.])
    error = torch.sum(torch.square(outputs-expected))
    
    return -error
