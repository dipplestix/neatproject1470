import torch
import sys
sys.path.append('../') # I added this because my imports were not working
from modified_flappy import run_model
from neat.make_phenotype import make_network

# load the best model and run it on flappy bird
# remember to change scale back to 1 in modified_flappy.py
best = torch.load('best.pt')
inputs = list(range(7))
outputs = list(range(7, 9))
m = make_network(best.gene_list,inputs, outputs, torch.nn.Softmax(1))
print(run_model(m))