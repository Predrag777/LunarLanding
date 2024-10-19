import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from sympy.unify.core import Variable
from torch.autograd import Variable
from collections import deque, namedtuple

class Brain(nn.Module):
    def __init__(self,state_size, action_size, seed=42):
        super(Brain, self).__init__()
        self.seed=torch.manual_seed(seed)
        self.fc1=nn.Linear(state_size, 64)       #Full-connected layer
        self.fc2=nn.Linear(64, 64)     #Full-connected layer
        self.output=nn.Linear(64,4)    #Output

    def forward(self, state):
        x=self.fc1(state)
        x=F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        return self.output(x)
