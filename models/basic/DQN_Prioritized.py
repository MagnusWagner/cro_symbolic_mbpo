import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import typing
import numpy as np
from models.utilities.ReplayBufferPrioritized import PrioritizedExperienceReplayBuffer, Experience, UniformReplayBuffer
import warnings

def synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module, tau: float = 1.0) -> None:
    """In place, synchronization of q_network_1 and q_network_2."""
    for target_param, local_param in zip(q_network_1.parameters(), q_network_2.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    # _ = q_network_1.load_state_dict(q_network_2.state_dict())


def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    _, actions = q_network(states).max(dim=1, keepdim=True)
    return actions




def evaluate_selected_actions(states: torch.Tensor,
                              actions: torch.Tensor,
                              rewards: torch.Tensor,
                              dones: torch.Tensor,
                              gamma: float,
                              q_network: nn.Module) -> torch.Tensor:
    """Compute the Q-values by evaluating the actions given the current states and Q-network."""
    next_q_values = q_network(states).gather(dim=1, index=actions)        
    q_values = rewards + (gamma * next_q_values * (1 - dones.int()))
    return q_values


def double_q_learning_update(states: torch.Tensor,
                             rewards: torch.Tensor,
                             dones: torch.Tensor,
                             gamma: float,
                             q_network_1: nn.Module,
                             q_network_2: nn.Module) -> torch.Tensor:
    """Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    actions = select_greedy_actions(states, q_network_1)
    q_values = evaluate_selected_actions(states, actions, rewards, dones, gamma, q_network_2)
    return q_values


def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            gamma: float,
                            q_network_1: nn.Module,
                            q_network_2: nn.Module) -> torch.Tensor:
    expected_q_values = double_q_learning_update(next_states, rewards, dones, gamma, q_network_1, q_network_2)
    q_values = q_network_1(states).gather(dim=1, index=actions)
    delta = expected_q_values - q_values
    return delta


A = typing.TypeVar('A', bound='Agent')


class Agent:
    
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError
        
    def learn(self, experiences: typing.List[Experience]) -> None:
        """Update the agent's state based on a collection of recent experiences."""
        raise NotImplementedError

    def save(self, filepath) -> None:
        """Save any important agent state to a file."""
        raise NotImplementedError
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplementedError


class DeepQAgent(Agent):

    def __init__(self,
                env,
                number_hidden_units: int,
                optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                batch_size: int,
                buffer_size: int,
                alpha: float,
                beta_annealing_schedule: typing.Callable[[int], float],
                epsilon_decay_schedule: typing.Callable[[int], float],
                gamma: float,
                random_state,
                tau: int = 1,
                model_buffer_flag = False,
                neighbour_flag = False,
                ) -> None:
        """
        Initialize a DeepQAgent.
        
        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        number_hidden_units (int): number of units in the hidden layers.
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        alpha (float): Strength of prioritized sampling; alpha >= 0.0.
        beta_annealing_schedule (callable): function that takes episode number and returns beta >= 0.
        epsilon_decay_schdule (callable): function that takes episode number and returns 0 <= epsilon < 1.
        alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        seed (int): random seed
        
        """

        self.env = env
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = None
        self._tau = tau
        
        # set seeds for reproducibility
        self._random_state = random_state
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "alpha": alpha,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._model_buffer_flag = model_buffer_flag
        if model_buffer_flag:
            self._model_memory = UniformReplayBuffer(
                batch_size = batch_size,
                buffer_size = buffer_size,
                random_state = self._random_state)
        if neighbour_flag:
            self._neighbour_memory = None
        self._memory = PrioritizedExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._beta_annealing_schedule = beta_annealing_schedule
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._gamma = gamma
        
        # initialize Q-Networks
        self._online_q_network = self._initialize_q_network(number_hidden_units)
        self._target_q_network = self._initialize_q_network(number_hidden_units)
        synchronize_q_networks(self._target_q_network, self._online_q_network)        
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)
        
        # initialize the optimizer
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0
        
    def _initialize_q_network(self, number_hidden_units: int) -> nn.Module:
        """Create a neural network for approximating the action-value function."""
        torch.manual_seed(self._random_state.get_state()[1][0])
        q_network = nn.Sequential(
            nn.Linear(in_features=self._state_size, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=self._action_size)
        )
        return q_network
           
    def _uniform_random_policy(self, state: torch.Tensor) -> int:
        """Choose an action uniformly at random."""
        return self._random_state.randint(self._action_size)
        
    def select_action(self,state, evaluation_flag = False):
        if evaluation_flag:
            with torch.no_grad():
                return self._online_q_network(state).argmax().view(1, 1).item()
        else:
            sample = self._random_state.random()
            self.eps_threshold = self._epsilon_decay_schedule(self._number_timesteps)
            # self.eps_threshold = 0.2
            if sample > self.eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    action_probs = self._online_q_network(state)
                    # print(action_probs)
                    action = action_probs.argmax().view(1, 1)
                    return action.item()
            else:
                return self.env.action_space.sample()
    
    def learn(self, idxs: np.array, experiences: np.array, sampling_weights: np.array, buffer_type = "real"):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = actions.long()
        # dones = dones.unsqueeze(dim=1)
        
        deltas = double_q_learning_error(states,
                                         actions,
                                         rewards,
                                         next_states,
                                         dones,
                                         self._gamma,
                                         self._online_q_network,
                                         self._target_q_network)
        avg_delta = deltas.mean()
        
        # update experience priorities
        priorities = (deltas.abs()
                            .cpu()
                            .detach()
                            .numpy()
                            .flatten())
        if buffer_type == "real":
            self._memory.update_priorities(idxs, priorities + 1e-6) # priorities must be positive!
            # compute the mean squared loss
            _sampling_weights = (torch.Tensor(sampling_weights)
                                    .view((-1, 1))).to(self._device)
            #TODO Switch to multiply loss with sampling weights again.
            loss = torch.mean((deltas * _sampling_weights)**2)
        elif buffer_type == "model":
            loss = torch.mean(deltas**2)
        elif buffer_type == "neighbour":
            _sampling_weights = (torch.Tensor(sampling_weights)
                        .view((-1, 1))).to(self._device)
            loss = torch.mean((deltas * _sampling_weights)**2)
        else:
            raise ValueError("buffer_type must be either real, model or neighbour.")

        
        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        synchronize_q_networks(self._target_q_network, self._online_q_network, tau=self._tau)
        return loss
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """
        Updates the agent's state based on feedback received from the environment.
        
        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.
        """
        if done:
            self._number_episodes += 1
            self._number_timesteps += 1
        else:
            self._number_timesteps += 1
        if next_state is None:
            next_state = torch.zeros_like(state).to(self._device)
        action = torch.tensor([action]).to(self._device)
        experience = Experience(state, action.view(1,1), reward.view(1,1), next_state, torch.tensor([done]).view(1,1))
        self._memory.add(experience)
        if len(self._memory)>=self._memory.batch_size:
            self.beta = self._beta_annealing_schedule(self._number_episodes)
            idxs, experiences, sampling_weights = self._memory.sample(self.beta)
            avg_delta = self.learn(idxs, experiences, sampling_weights)
            return avg_delta
        return 0.0
    
    def add_to_model_replay_buffer(self,
             states: np.array,
             actions: int,
             rewards: float,
             next_states: np.array,
             dones: bool) -> None:
        """
        Updates the agent's state based on feedback received from the environment.
        
        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.
        """
        for i in range(states.shape[0]):
            if next_states[i] is None:
                warnings.warn("add_to_model_replay_buffer: next_state is None")
                next_state = torch.zeros_like(states[i]).to(self._device)
            else:
                next_state = next_states[i]
            experience = Experience(states[i], actions[i].view(1,1), rewards[i].view(1,1), next_state, dones[i].view(1,1))
            self._model_memory.add(experience)


    def add_neighbour_buffer(self, neighbour_replay_buffer) -> None:
        self._neighbour_memory = neighbour_replay_buffer

    # def learn_from_buffer(self):
    #     """Update the agent's state based on a collection of recent simulated experiences."""
    #     if len(self._model_memory)>=self._model_memory.batch_size:
    #         self.beta = self._beta_annealing_schedule(self._number_episodes)
    #         idxs, experiences, sampling_weights = self._model_memory.sample(self.beta)
    #         avg_delta = self.learn(idxs, experiences, sampling_weights, buffer_type = "model")
    #         return avg_delta
    #     return 0.0
    def learn_from_buffer(self):
        """Update the agent's state based on a collection of recent simulated experiences."""
        if len(self._model_memory)>=self._model_memory.batch_size:
            idxs, experiences = self._model_memory.uniform_sample(replace = True)
            avg_delta = self.learn(idxs, experiences, None, buffer_type = "model")
            return avg_delta
        return 0.0

    def learn_from_neighbour_buffer(self, pretrain_flag = False):
        """Update the agent's state based on a collection of recent simulated experiences."""
        if pretrain_flag:
            num_pretraining_steps = len(self._neighbour_memory)*10//(self._neighbour_memory.batch_size)
            avg_deltas = []
            for i in range(num_pretraining_steps):
                idxs, experiences, weights = self._neighbour_memory.sample_neighbour_experience()
                avg_delta = self.learn(idxs, experiences, weights, buffer_type = "neighbour")
                avg_deltas.append(avg_delta)
            return avg_deltas
        if len(self._neighbour_memory)>=self._neighbour_memory.batch_size:
            idxs, experiences, weights = self._neighbour_memory.sample_neighbour_experience()
            avg_delta = self.learn(idxs, experiences, weights, buffer_type = "neighbour")
            return avg_delta
        else:
            warnings.warn("Neighbour memory is not large enough.")
        return 0.0
    

    def save(self, filepath: str) -> None:
        """
        Saves the state of the DeepQAgent.
        
        Parameters:
        -----------
        filepath (str): filepath where the serialized state should be saved.
        
        Notes:
        ------
        The method uses `torch.save` to serialize the state of the q-network, 
        the optimizer, as well as the dictionary of agent hyperparameters.
        
        """
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "alpha": self._memory.alpha,
                "beta_annealing_schedule": self._beta_annealing_schedule,
                "batch_size": self._memory.batch_size,
                "buffer_size": self._memory.buffer_size,
                "epsilon_decay_schedule": self._epsilon_decay_schedule,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency
            }
        }
        torch.save(checkpoint, filepath)
