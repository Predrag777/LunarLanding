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
from network import Brain

class ReplayMemory(object):
    def __init__(self, capacity):
        self.device=torch.devicee("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity=capacity
        self.memory=[]

    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]