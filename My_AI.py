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

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step=(self.t_step+1)%4
        if self.t_step==0:  #if t_step is divisible with 4
            if len(self.memory.memory):
                experiences=self.memory.sample(100)
                self.learn(experiences, 0.99)   #gamma=0.99
    
    def act(self, state, epsilon=0.):
        state=torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()

        with torch.no_grad():   #State of inference not of training. Model makes predictions over data whcih did not be seen before
            action_values=self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random()>epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, next_states, actions, rewards,dones=experiences
        next_q_targets=self.target_qnetwork(next_states).detache().max(1)[0].unsqueeze(1)
        q_targets=rewards+(gamma*next_q_targets*(1-dones))
        q_expected=self.local_qnetwork(states).gather(1,actions)
        loss=F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter*local_param.data+(1.0-interpolation_parameter)*target_param.data)
