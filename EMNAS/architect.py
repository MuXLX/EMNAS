import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)


    def step(self,input_valid, target_valid,encoding,V,E=0):
        self.optimizer.zero_grad()
        loss = self.model._loss(input_valid, target_valid,encoding,V,E)
        loss.backward()
        self.optimizer.step()
