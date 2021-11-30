from neat.neat_structures import Genome, Gene
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List
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


def make_netowrk(gen: Geome):
    fwd, rev = make_dics(gen.directedConnects)
    n_hidden = max([len(fwd[n]) for n in fwd])
    output = [n for n in gen.nodes if n not in fwd.keys()]

    last_layer = rev[output]
    torch.nn.Linear
    layers = [last_layer]


