import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import typing
import numpy as np
import clingo
from simulation_env.environment_maincrops.clingo_strings import program_start, program_end
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module) -> None:
    """In place, synchronization of q_network_1 and q_network_2."""
    _ = q_network_1.load_state_dict(q_network_2.state_dict())


def select_greedy_actions(states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    action_values = q_network(states)
    # filter_mask = filter_mask.bool()
    # action_values[filter_mask] = -5.0
    _, actions = action_values.max(dim=1, keepdim=True)

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
                            #  filter_mask: torch.Tensor,
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
                            actions_allowed: torch.Tensor,
                            gamma: float,
                            q_network_1: nn.Module,
                            q_network_2: nn.Module) -> torch.Tensor:

    # # print average reward over rewards where actions_allowed == 1.0
    # print(rewards[actions_allowed == 1.0].mean())
    # print(rewards[actions_allowed == 0.0].mean())
    expected_q_values = double_q_learning_update(next_states, rewards, dones, gamma, q_network_1, q_network_2)
    # print(expected_q_values[actions_allowed == 1.0].mean())
    # print(expected_q_values[actions_allowed == 0.0].mean())

    q_values = q_network_1(states).gather(dim=1, index=actions)
    # print(expected_q_values)
    # print(q_values)
    delta = expected_q_values - q_values
    return delta

# def filter_actions(filter_information):
#         control = clingo.Control()
#         week = filter_information[0]
#         ground_type = filter_information[1]
#         drywet = filter_information[2]
#         previous_crops_selected = filter_information[3:]

#         # Example:
#         # ground_type_info(0).
#         # drywet_info(0).
#         # week_info(24).

#         # previous_actions_info(-5,6).
#         # previous_actions_info(-4,1).
#         # previous_actions_info(-3,8).
#         # previous_actions_info(-2,9).
#         # previous_actions_info(-1,1).
#         configuration_string = f"""
#         week_info({int(week)}).
#         ground_type_info({int(ground_type)}).
#         drywet_info({int(drywet)}).
#         """
#         for i, crop in enumerate(previous_crops_selected):
#             flag = False
#             if crop and crop != -1.0:
#                 flag = True
#                 configuration_string += f"""previous_actions_info({i-5},{int(crop)}).\n"""
#             if not flag:
#                 configuration_string += f"""previous_actions_info({-1},{-1}).\n"""
#         program = program_start + configuration_string + program_end["only_break_rules_and_timing"]

#         control.add("base", [], program)
#         control.ground([("base", [])])

#         # Solve the program and print the immediate candidate actions
#         solutions = []
#         def on_model(model):
#             solutions.append(model.symbols(shown=True))
#         control.solve(on_model=on_model)
#         if len(solutions) > 0:
#             solution = solutions[0]
#             possible_actions = [symbol.arguments[0].number for symbol in solution]
#             if len(possible_actions) > 0:
#                 return possible_actions
#             else:
#                 return None
#         else:
#             return None
# def calculate_action_probabilities_for_greedy_policy(action_probabilities_array, filter_informations):
#     for i in range(action_probabilities_array.shape[0]):
#         filter_information = filter_informations[0].cpu()
#         possible_actions = filter_actions(filter_information)
#         if possible_actions:
#             possible_actions_tensor = torch.tensor(possible_actions, device=DEVICE, dtype=torch.int)
#         else:
#             print("No possible actions.")
#         if possible_actions:
#             action_probabilities = action_probabilities_array[i].cpu()
#             mask = torch.ones_like(action_probabilities, dtype=bool)
#             mask[possible_actions_tensor] = False
#             action_probabilities[mask] = 0.0
#             action_probabilities_array[i]=action_probabilities.to(DEVICE)
#     return action_probabilities_array

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "action_allowed"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.
        
        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0 # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size
    
    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def add(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0
    
    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size
    
    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]        
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        
        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities



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
             done: bool,
             filter_information) -> None:
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
                delta_decay_schedule: typing.Callable[[int], float],
                gamma: float,
                update_frequency: int = 1,
                seed: int = None,
                rule_options = "all", # "all" or "only_break_rules_and_timing"
                only_filter = False
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
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        seed (int): random seed
        
        """
        self.env = env
        self.rule_options = rule_options
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = None
        self.control = clingo.Control()
        self.only_filter = only_filter
        
        # set seeds for reproducibility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
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
        self._memory = PrioritizedExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._beta_annealing_schedule = beta_annealing_schedule
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._delta_decay_schedule = delta_decay_schedule
        self._gamma = gamma
        
        # initialize Q-Networks
        self._update_frequency = update_frequency
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
        
    # def _greedy_policy(self, state: torch.Tensor) -> int:
    #     """Choose an action that maximizes the action_values given the current state."""
    #     actions = select_greedy_actions(state, self._online_q_network)
    #     action = (actions.cpu()  # actions might reside on the GPU!
    #                      .item())
    #     return action
    
    # def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float) -> int:
    #     """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
    #     if self._random_state.random() < epsilon:
    #         action = self._uniform_random_policy(state)
    #     else:
    #         action = self._greedy_policy(state)
    #     return action
    
    def filter_actions(self, filter_information):
        self.control = clingo.Control()
        week = filter_information[0]
        ground_type = filter_information[1]
        drywet = filter_information[2]
        previous_crops_selected = filter_information[3:]

        # Example:
        # ground_type_info(0).
        # drywet_info(0).
        # week_info(24).

        # previous_actions_info(-5,6).
        # previous_actions_info(-4,1).
        # previous_actions_info(-3,8).
        # previous_actions_info(-2,9).
        # previous_actions_info(-1,1).
        configuration_string = f"""
        week_info({int(week)}).
        ground_type_info({int(ground_type)}).
        drywet_info({int(drywet)}).
        """
        for i, crop in enumerate(previous_crops_selected):
            flag = False
            if crop and crop != -1.0:
                flag = True
                configuration_string += f"""previous_actions_info({i-5},{int(crop)}).\n"""
            if not flag:
                configuration_string += f"""previous_actions_info({-1},{-1}).\n"""
        program = program_start + configuration_string + program_end[self.rule_options]

        self.control.add("base", [], program)
        self.control.ground([("base", [])])

        # Solve the program and print the immediate candidate actions
        solutions = []
        def on_model(model):
            solutions.append(model.symbols(shown=True))
        self.control.solve(on_model=on_model)
        if len(solutions) > 0:
            solution = solutions[0]
            possible_actions = [symbol.arguments[0].number for symbol in solution]
            if len(possible_actions) > 0:
                return possible_actions
            else:
                return None
        else:
            return None
        
    # def filter_actions_for_greedy_policy(self, action_probabilities, filter_informations):
    #     self.control = clingo.Control()
    #     for i in range(action_probabilities.shape[0]):
    #         week = filter_informations[i][0].cpu()
    #         ground_type = filter_informations[i][1].cpu()
    #         drywet = filter_informations[i][2].cpu()
    #         previous_crops_selected = filter_informations[i][3:].cpu()
    #         configuration_string = f"""
    #         week_info({int(week)}).
    #         ground_type_info({int(ground_type)}).
    #         drywet_info({int(drywet)}).
    #         """
    #         for i, crop in enumerate(previous_crops_selected):
    #             flag = False
    #             if crop:
    #                 flag = True
    #                 configuration_string += f"""previous_actions_info({i-5},{crop}).\n"""
    #             if not flag:
    #                 configuration_string += f"""previous_actions_info({-1},{-1}).\n"""
    #         program = program_start + configuration_string + program_end[self.rule_options]
    #         self.control.add("base", [], program)
    #         self.control.ground([("base", [])])

    #         # Solve the program and print the immediate candidate actions
    #         solutions = []
    #         def on_model(model):
    #             solutions.append(model.symbols(shown=True))
    #         self.control.solve(on_model=on_model)
    #         if len(solutions) > 0:
    #             solution = solutions[0]
    #             possible_actions = [symbol.arguments[0].number for symbol in solution]
    #             if len(possible_actions) > 0:
    #                 return possible_actions
    #             else:
    #                 return None
    #         else:
    #             return None                

    def select_action(self, state, filter_information, greedy = False):
        sample = random.random()
        sample_delta = random.random()
        self.eps_threshold = self._epsilon_decay_schedule(self._number_timesteps)
        self.delta_threshold = self._delta_decay_schedule(self._number_timesteps)
        filtered_flag = False
        # TODO Remove
        # self.delta_threshold = 0.9
        possible_actions = self.filter_actions(filter_information)
        if possible_actions:
            possible_actions_tensor = torch.tensor(possible_actions, device=self._device, dtype=torch.int)
        else:
            print("No possible actions.")
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_values = self._online_q_network(state)
            if (sample_delta <= self.delta_threshold or greedy) and possible_actions :
                mask = torch.ones_like(action_values, dtype=bool)
                mask[0][possible_actions_tensor] = False
                action_values[mask] = -np.inf
                filtered_flag = True
            if (sample > self.eps_threshold or greedy) and not self.only_filter:
                selected_action = action_values.argmax().view(1, 1)
                if possible_actions and selected_action.item() in possible_actions:
                    filtered_flag = True
                return selected_action, filtered_flag
            else:
                if (sample_delta <= self.delta_threshold or self.only_filter) and possible_actions:
                    return torch.tensor([random.choice(possible_actions)], device=self._device, dtype=torch.long), True
                else:
                    selected_action = self.env.action_space.sample()
                    if possible_actions and selected_action in possible_actions:
                        filtered_flag = True
                    return torch.tensor([selected_action], device=self._device, dtype=torch.long), filtered_flag
        
    def select_greedy_action(self,state):
        # self.eps_threshold = 0.2
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self._online_q_network(state).argmax().view(1, 1)
    
    def learn(self, idxs: np.array, experiences: np.array, sampling_weights: np.array):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones, actions_allowed = (torch.stack(vs,0).squeeze(1).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = actions.long()
        # dones = dones.unsqueeze(dim=1)
        
        deltas = double_q_learning_error(states,
                                         actions,
                                         rewards,
                                         next_states,
                                         dones,
                                         actions_allowed,
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
        self._memory.update_priorities(idxs, priorities + 1e-6) # priorities must be positive!
        
        # compute the mean squared loss
        _sampling_weights = (torch.Tensor(sampling_weights)
                                  .view((-1, 1))).to(self._device)
        loss = torch.mean((deltas * _sampling_weights)**2)
        
        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        synchronize_q_networks(self._target_q_network, self._online_q_network)
        return loss
    
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
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool,
             filter_information,
             env) -> None:
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
            next_state = torch.ones(len(state[0])).view(1,194).to(self._device)
        possible_actions = self.filter_actions(filter_information)
        filter_mask = torch.zeros(env.action_space.n)
        if possible_actions:
            filter_mask[possible_actions] = 1.0
        filter_mask = filter_mask.to(self._device)
        action_allowed = filter_mask[action].view(1,1)
        action = action.view(1)
        experience = Experience(state, action.view(1,1), reward.view(1,1), next_state, torch.tensor([done]).view(1,1), action_allowed)
        self._memory.add(experience)
        if len(self._memory)>=self._memory.batch_size:
            self.beta = self._beta_annealing_schedule(self._number_episodes)
            idxs, experiences, sampling_weights = self._memory.sample(self.beta)
            avg_delta = self.learn(idxs, experiences, sampling_weights)
            return avg_delta
        return 0.0



