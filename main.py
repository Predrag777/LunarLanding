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
from Experience import ReplayMemory
from My_AI import Agent
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



class ReplayMemory(object):
    def __init__(self, capacity):
        self.device=torch.devicee("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity=capacity
        self.memory=[]

    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences=random.sample(self.memory, k=batch_size)
        states=torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions=torch.from_numpy(np.vstack(e[1] for e in experiences if e is not None)).float().to(self.device)
        rewards=torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states=torch.from_numpy(np.vstack(e[3] for e in experiences if e is not None)).float().to(self.device)
        dones=torch.from_numpy(np.vstack(e[3] for e in experiences if e is not None)).astype(np.uint8).float().to(self.device)

        return states, next_states, actions, rewards, dones







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




agent=Agent(state_size, number_action)



number_episodes=2000
maximum_number_timesteps_per_episode=1000
epsilon_starting_value=1.0
epsilon_ending_value=0.01
epsilon_decay_value=0.995
epsilon=epsilon_starting_value
score_on_100_episodes=deque(maxlen=100)


for episode in range(1,number_episodes+1):
    state,_=env.reset()
    score=0
    for t in range(maximum_number_timesteps_per_episode):
        action=agent.act(state, t)
        next_state, reward, done, _, _=env.step(action)
        agent.step(state,action,reward,next_state,done)
        state=next_state
        score+=reward
        if done:
            break
    score_on_100_episodes.append(score)
    epsilon=max(epsilon_ending_value, epsilon_decay_value*epsilon)
