from copy import deepcopy
import torch
from torch import nn


def make_dics(connections):
    fwd = {}
    rev = {}
    for c in connections:
        i = c[0]
        o = c[1]
        try:
            fwd[i].append(o)
        except KeyError:
            fwd[i] = [o]
        try:
            rev[o].append(i)
        except KeyError:
            rev[o] = [i]
    return fwd, rev


def make_network(gen: GeneList, actual_inputs, actual_outputs, last_layer_fun=torch.sigmoid):
    # Find what's input and output
    fwd, rev = make_dics(gen.active_connects.keys())
    outputs = actual_outputs
    inputs = actual_inputs

    # Record what goes into each layer, the matrix, and what comes out of each layer
    layers = []
    in_ls = []
    out_ls = []
    nn_layers = []

    # Go backwards and determine the network graph
    nodes_to_process = deepcopy(outputs)
    max_node = max(gen.nodes)
    while nodes_to_process:
        next_layer = []
        for n in nodes_to_process:
            try:
                next_layer.extend(rev[n])
            except KeyError:
                pass
        layers.append(list(set(next_layer)))
        nodes_to_process = next_layer
    layers.reverse()
    layers = layers[1:]
    layers.append(outputs)

    # Build the model
    for i in range(len(layers) - 1):
        # if i == 0:
        #     in_l = inputs
        # else:
        in_l = layers[i]
        if i == len(layers) - 1:
            out_l = outputs
        else:
            out_l = layers[i + 1]
        out_l = [n for n in out_l if n not in inputs]
        nn_layer = nn.Linear(len(in_l), len(out_l), bias=False)
        for j, ni in enumerate(in_l):
            for k, no in enumerate(out_l):
                if ni == no:
                    w = 1
                else:
                    try:
                        w = gen.active_connects[(ni, no)]
                    except KeyError:
                        w = 0
                with torch.no_grad():
                    nn_layer.weight[k, j] = w
        nn_layers.append(nn_layer)
        in_ls.append(in_l)
        out_ls.append(out_l)

    # We output a model that inherits a lot of stuff from the function above
    def model(val_in):
        if len(val_in.shape) == 1:
            val_in = torch.reshape(val_in, [1, -1])
        assert len(val_in.shape) == 2, 'Requires 2D Tensor'
        values = torch.zeros(val_in.shape[0], max_node + 1)
#         print(f'layers: {layers}')
#         print(f'in_ls: {in_ls}')
#         print(f'out_ls: {out_ls}')

#         [print(g) for g in gen.genes]
        values[:, actual_inputs]  = val_in

        for m, layer in enumerate(nn_layers):
            ins = values[:, in_ls[m]]
            loop_value = layer(ins)
            if m == len(nn_layers) - 1:
                loop_value = last_layer_fun(loop_value)
            else:
                loop_value = torch.sigmoid(loop_value)
            values[:, out_ls[m]] += loop_value

        return values[:, actual_outputs]

    return model
