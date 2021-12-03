import torch


def get_fitness(model):
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    outputs = model(inputs)
    expected = torch.tensor([0., 1., 1., 0.])
    print(outputs - expected)
    print(torch.square(outputs-expected))
    error = torch.sum(torch.square(outputs-expected))
    
    return -error