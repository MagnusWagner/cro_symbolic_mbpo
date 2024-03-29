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
from models.utilities.ReplayBufferPrioritized import PrioritizedExperienceReplayBufferSymbolic, Experience_Symbolic, UniformReplayBuffer
import clingo
from simulation_env.environment_maincrops.clingo_strings import program_start, program_end
import warnings


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module, tau: float = 1.0) -> None:
    """In place, synchronization of q_network_1 and q_network_2."""
    for target_param, local_param in zip(q_network_1.parameters(), q_network_2.parameters()):
        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    # _ = q_network_1.load_state_dict(q_network_2.state_dict())


# TODO Change to set non-allowed-actions to -np.inf
def select_next_greedy_actions(next_states: torch.Tensor, q_network: nn.Module, next_filter_masks: torch.Tensor) -> torch.Tensor:
    """Select the greedy action for the current state given some Q-network."""
    next_action_values = q_network(next_states)
    next_filter_masks = next_filter_masks.bool()
    next_action_values[next_filter_masks] = -np.inf
    _, next_actions = next_action_values.max(dim=1, keepdim=True)
    return next_actions




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


def double_q_learning_update(next_states: torch.Tensor,
                            rewards: torch.Tensor,
                            dones: torch.Tensor,
                            gamma: float,
                            q_network_1: nn.Module,
                            q_network_2: nn.Module,
                            next_filter_masks: torch.Tensor,) -> torch.Tensor:
    """Double Q-Learning uses Q-network 1 to select actions and Q-network 2 to evaluate the selected actions."""
    next_actions = select_next_greedy_actions(next_states, q_network_1, next_filter_masks)
    q_values = evaluate_selected_actions(next_states, next_actions, rewards, dones, gamma, q_network_2)
    return q_values


def double_q_learning_error(states: torch.Tensor,
                            actions: torch.Tensor,
                            rewards: torch.Tensor,
                            next_states: torch.Tensor,
                            dones: torch.Tensor,
                            next_filter_masks: torch.Tensor,
                            gamma: float,
                            q_network_1: nn.Module,
                            q_network_2: nn.Module) -> torch.Tensor:

    # # print average reward over rewards where actions_allowed == 1.0
    # print(rewards[actions_allowed == 1.0].mean())
    # print(rewards[actions_allowed == 0.0].mean())
    expected_q_values = double_q_learning_update(next_states, rewards, dones, gamma, q_network_1, q_network_2, next_filter_masks)
    # print(expected_q_values[actions_allowed == 1.0].mean())
    # print(expected_q_values[actions_allowed == 0.0].mean())

    q_values = q_network_1(states).gather(dim=1, index=actions)
    # print(expected_q_values)
    # print(q_values)
    delta = expected_q_values - q_values
    return delta


A = typing.TypeVar('A', bound='Agent')


class Agent:
    
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError
        
    def learn(self, experiences: typing.List[Experience_Symbolic]) -> None:
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
                random_state,
                tau: int = 1,
                rule_options = "all", # "all" or "only_break_rules_and_timing"
                only_filter = False,
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
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        seed (int): random seed
        
        """
        self.env = env
        self._state_size = env.observation_space.shape[0]
        self._action_size = env.action_space.n
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = None
        self.control = clingo.Control()
        self.only_filter = only_filter
        self.rule_options = rule_options
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
        self._memory = PrioritizedExperienceReplayBufferSymbolic(**_replay_buffer_kwargs)
        self._beta_annealing_schedule = beta_annealing_schedule
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._delta_decay_schedule = delta_decay_schedule
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
        
    def filter_actions(self, filter_information):
        self.control = clingo.Control()
        week = filter_information[0]
        ground_type = filter_information[1]
        drywet = filter_information[2]
        humus_level = int(filter_information[3]*1000)
        humus_minimum_level = int(filter_information[4]*1000)
        previous_crops_selected = filter_information[5:]
        # Example:
        # ground_type_info(0).
        # drywet_info(0).
        # week_info(24).
        # humus_info(2400,2000)

        # previous_actions_info(-5,6).
        # previous_actions_info(-4,1).
        # previous_actions_info(-3,8).
        # previous_actions_info(-2,9).
        # previous_actions_info(-1,1).
        configuration_string = f"""
        week_info({int(week)}).
        ground_type_info({int(ground_type)}).
        drywet_info({int(drywet)}).
        humus_info({humus_level},{humus_minimum_level}).
        """
        for i, crop in enumerate(previous_crops_selected):
            flag = False
            if crop is not None and crop != -1.0:
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
        
              

    def select_action(self, state, filter_information, evaluation_flag = False):
        sample = self._random_state.random()
        sample_delta = self._random_state.random()
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
            if (sample_delta <= self.delta_threshold or evaluation_flag) and possible_actions :
                mask = torch.ones_like(action_values, dtype=bool)
                mask[0][possible_actions_tensor] = False
                action_values[mask] = -np.inf
                filtered_flag = True
            if (sample > self.eps_threshold or evaluation_flag) and not self.only_filter:
                selected_action = action_values.argmax().view(1, 1)
                if possible_actions and selected_action.item() in possible_actions:
                    filtered_flag = True
                return selected_action.item(), filtered_flag
            else:
                if (sample_delta <= self.delta_threshold or self.only_filter) and possible_actions:
                    # return torch.tensor([self._random_state.choice(possible_actions)], device=self._device, dtype=torch.long), True
                    return self._random_state.choice(possible_actions), True
                else:
                    selected_action = self.env.action_space.sample()
                    if possible_actions and selected_action in possible_actions:
                        filtered_flag = True
                    # return torch.tensor([selected_action], device=self._device, dtype=torch.long), filtered_flag
                    return selected_action, filtered_flag
    
    def learn(self, idxs: np.array, experiences: np.array, sampling_weights: np.array, buffer_type = "real"):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones, next_filter_masks = (torch.stack(vs,0).squeeze(1).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = actions.long()
        # dones = dones.unsqueeze(dim=1)
        
        deltas = double_q_learning_error(states,
                                         actions,
                                         rewards,
                                         next_states,
                                         dones,
                                         next_filter_masks,
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
            _sampling_weights = (torch.Tensor(sampling_weights)
                                    .view((-1, 1))).to(self._device)
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
             done: bool,
             next_filter_information,
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
        filter_information (numpy.array): information about the current state of the environment.
        env (gymnasium.Environment): the environment in which the agent is acting.
        """
        if done:
            self._number_episodes += 1
            self._number_timesteps += 1
        else:
            self._number_timesteps += 1
        if next_state is None:
            next_state = torch.zeros_like(state).to(self._device)
        action = torch.tensor([action]).to(self._device)
        next_possible_actions = self.filter_actions(next_filter_information)
        next_filter_mask = torch.zeros(env.action_space.n)
        if next_possible_actions is not None and not done:
            next_filter_mask = torch.ones(env.action_space.n)
            next_filter_mask[next_possible_actions] = 0.0 # TODO Check if this is correct
        next_filter_mask = next_filter_mask.to(self._device)
        # action_allowed = filter_mask[action].view(1,1)
        experience = Experience_Symbolic(state, action.view(1,1), reward.view(1,1), next_state, torch.tensor([done]).view(1,1), next_filter_mask)
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
             dones: bool,
             next_filter_informations,
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
        next_filter_information (numpy.array): information about the current state of the environment.
        """
        for i in range(states.shape[0]):
            if next_states[i] is None:
                warnings.warn("add_to_model_replay_buffer: next_state is None")
                next_state = torch.zeros_like(states[i]).to(self._device)
            else:
                next_state = next_states[i]
            next_possible_actions = self.filter_actions(next_filter_informations[i])
            next_filter_mask = torch.zeros(env.action_space.n)
            if next_possible_actions is not None and not dones[i]:
                next_filter_mask = torch.ones(env.action_space.n)
                next_filter_mask[next_possible_actions] = 0.0 # TODO Check if this is correct
            next_filter_mask = next_filter_mask.to(self._device)
            # action_allowed = filter_mask[action].view(1,1)
            experience = Experience_Symbolic(states[i], actions[i].view(1,1), rewards[i].view(1,1), next_state, dones[i].view(1,1), next_filter_mask)
            self._model_memory.add(experience)

    def add_neighbour_buffer(self, neighbour_replay_buffer) -> None:
        self._neighbour_memory = neighbour_replay_buffer
    
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
