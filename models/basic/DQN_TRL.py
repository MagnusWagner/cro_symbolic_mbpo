# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.policy import MaxBoltzmannQPolicy
# from rl.policy import EpsGreedyQPolicy
# from rl.policy import GreedyQPolicy
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy
# import tensorflow

import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import tensorflow.keras as keras

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNModel(nn.Module):
    def __init__(self, env, seed = 42):
        super(DQNModel, self).__init__()
        self.states = env.observation_space
        self.actions = env.action_space
        n_observations = self.states.shape[0]
        n_actions = self.actions.n
        
        self.layer1 = nn.Linear(n_observations, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent():
    def __init__(self, env, seed = 42):
        super(DQNAgent, self).__init__()
        self.BATCH_SIZE = 1
        self.GAMMA = 0.99
        self.EPS_START = 0.4
        self.EPS_END = 0.05
        self.EPS_DECAY = 100
        self.TAU = 0.05
        self.LR = 1e-4
        self.steps_done = 0
        self.memory = ReplayMemory(10000)
        self.env = env
        # Get number of actions from gym action space
        self.n_actions = env.action_space.n
        # Get the number of state observations
        self.n_observations = env.observation_space.shape[0]

        self.policy_net = DQNModel(self.env).to(device)
        self.target_net = DQNModel(self.env).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        self.eps_threshold = 0.2



    def select_action(self,state):
        sample = random.random()
        # self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.eps_threshold = 0.1
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                test_state = self.policy_net(state)
                test_max_state = test_state.argmax()
                test_view = test_max_state.view(1, 1)
                return test_view
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)
        
    def optimize_model(self):
        transition = self.memory.sample(1)[0]
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                     batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                             if s is not None])
        state = transition.state
        action = transition.action
        reward = transition.reward
        next_state = transition.next_state

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_value = self.policy_net(state)[action]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        if next_state:
            with torch.no_grad():
                next_state_value = self.target_net(state).max()
        else:
            next_state_value = 0.0

        # Compute the expected Q values
        expected_state_action_value = (next_state_value * self.GAMMA) + reward

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_value.flatten(), expected_state_action_value.flatten())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # def optimize_model(self):
    #     if len(self.memory) < self.BATCH_SIZE:
    #         return
    #     transitions = self.memory.sample(self.BATCH_SIZE)
    #     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    #     # detailed explanation). This converts batch-array of Transitions
    #     # to Transition of batch-arrays.
    #     batch = Transition(*zip(*transitions))

    #     # Compute a mask of non-final states and concatenate the batch elements
    #     # (a final state would've been the one after which simulation ended)
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                         batch.next_state)), device=device, dtype=torch.bool)
    #     non_final_next_states = torch.cat([s for s in batch.next_state
    #                                                 if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)

    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken. These are the actions which would've been taken
    #     # for each batch state according to policy_net
    #     state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    #     # Compute V(s_{t+1}) for all next states.
    #     # Expected values of actions for non_final_next_states are computed based
    #     # on the "older" target_net; selecting their best reward with max(1)[0].
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
    #     with torch.no_grad():
    #         next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
    #     # Compute the expected Q values
    #     expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

    #     # Compute Huber loss
    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    #     # Optimize the model
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     # In-place gradient clipping
    #     torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    #     self.optimizer.step()