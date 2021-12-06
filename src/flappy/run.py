import sys
sys.path.append('../') # I added this because my imports were not working

from neat.neat_structures import *
from copy import deepcopy
from random import random, choice, uniform, sample
from typing import Callable, List
import torch
from torch import nn
from neat.neat_functions import *
from modified_flappy import *

def fitness_12(inputs):
    return 12

pop = initialization(7, 2, fitness_12, 10)

m = make_network(pop[0].gene_list, torch.nn.Softmax(1))

score = run_model(m)

print(f"SCORE: {score}")
