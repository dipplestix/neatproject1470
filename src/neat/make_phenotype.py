from neat.neat_structures import Genome, Gene
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List
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


def make_network(gen: Genome):
    fwd, rev = make_dics(gen.active_connects.keys())
    outputs = [n for n in gen.nodes if n not in fwd.keys()]
    inputs = [n for n in gen.nodes if n not in rev.keys()]
    gaps = max(outputs) - max(inputs)
    layers = []
    nn_layers = []
    nodes_to_process = deepcopy(outputs)
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
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            if i == 0:
                in_l = inputs
            else:
                in_l = out_l
            out_l = deepcopy(in_l)
            out_l.extend(layers[i+1])
            out_l = list(set(out_l))
            # print(in_l)
            # print(out_l)
            nn_layer = nn.Linear(len(in_l), len(out_l), bias=False)
            for ni in in_l:
                for no in out_l:
                    in_node = ni
                    out_node = no
                    if ni == no:
                        w = 1
                    elif no in in_l:
                        w = 0
                    else:
                        try:
                            w = gen.active_connects[(ni, no)]
                        except KeyError:
                            w = 0
                    if ni not in inputs:
                        in_node -= gaps
                    if no not in inputs:
                        out_node -= gaps
                    with torch.no_grad():
                        nn_layer.weight[out_node, in_node] = w
            # print(nn_layer.weight)
            nn_layers.append(nn_layer)
            nn_layers.append(nn.ReLU(inplace=True))
    else:
        out_l = layers[0]
    last_layer = nn.Linear(len(out_l), len(outputs), bias=False)
    for i in out_l:
        for o in outputs:
            in_node = i
            try:
                w = gen.active_connects[(i, o)]
            except KeyError:
                w = 0
            if i not in inputs:
                in_node -= gaps
            with torch.no_grad():
                last_layer.weight[o-min(outputs), in_node] = w
    # print(last_layer.weight)
    nn_layers.append(last_layer)
    # print(layers)
    # print(nn_layers)
    model = nn.Sequential(*nn_layers)
    return model
    
            
        
    


