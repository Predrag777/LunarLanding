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

learning_rate=5e-4

import gymnasium as gym
env=gym.make('LunarLander-v3')
state_shape=env.observation_space.shape
state_size=env.observation_space.shape[0]
number_action=env.action_space.n
print(number_action)


miibatch_size=100
gama=0.99                   #discount factor
replay_buffer_size=int(1e5) #memory of the agent
interpolation_parameter=1e-3 #NE ZNAM STA JE OVO

print("SS")