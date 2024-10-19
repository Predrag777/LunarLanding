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

from DEEP_Q_LEARNING.main import replay_buffer_size
from network import Brain
from Experience import ReplayMemory

class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.devicee("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size=state_size
        self.action_size=action_size
        self.local_qnetwork=Brain(state_size, action_size).to(self.device)
        self.target_qnetwork=Brain(state_size, action_size).to(self.device)
        self.optimizer=optim.Adam(self.local_qnetwork.parameters(), lr=5e-1)
        self.memory=ReplayMemory(replay_buffer_size)
        self.t_step=0

