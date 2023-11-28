import numpy as np
import torch
import typing
from models.utilities.Network import Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from numpy import random
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
class SACAgent:

    def __init__(self, 
                env,
                number_hidden_units: int,
                critic_optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                actor_optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                batch_size: int,
                buffer_size: int,
                prio_alpha: float,
                temperature_initial: float,
                beta_annealing_schedule: typing.Callable[[int], float],
                delta_decay_schedule: typing.Callable[[int], float],
                tau:float,
                gamma: float,
                temperature_decay_schedule: typing.Callable[[int], float],
                random_state,
                rule_options = "all", # "all" or "only_break_rules_and_timing"
                only_filter = False,
                model_buffer_flag = False,
                neighbour_flag = False,
                 ):
        self._random_state = random_state
        self._random_state2 = np.random.RandomState(self._random_state.get_state()[1][0]+1)
        self._random_state3 = np.random.RandomState(self._random_state.get_state()[1][0]+2)
        self._random_state4 = np.random.RandomState(self._random_state.get_state()[1][0]+3)
        self._env = env
        self.rule_options = rule_options
        self._number_hidden_units = number_hidden_units
        self._gamma = gamma
        self._default_tau = tau
        self._temperature_initial = temperature_initial
        self._beta_annealing_schedule = beta_annealing_schedule
        self._delta_decay_schedule = delta_decay_schedule
        self._temperature_decay_schedule = temperature_decay_schedule
        self.beta = None
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control = clingo.Control()
        self.only_filter = only_filter
        self._critic_local = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units,
                                    random_state = self._random_state).to(self._device)
        self._critic_local2 = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units,
                                    random_state = self._random_state2).to(self._device)
        
        self._critic_optimizer = critic_optimizer_fn(self._critic_local.parameters())
        self._critic_optimizer2 = critic_optimizer_fn(self._critic_local2.parameters())

        self._critic_target = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units,
                                    random_state = self._random_state3).to(self._device)
        self._critic_target2 = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units,
                                    random_state = self._random_state4).to(self._device)
        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0

        self.soft_update_target_networks(tau=1.)

        self._actor_local = Network(
            input_dimension=self._state_size,
            output_dimension=self._action_size,
            number_hidden_units=self._number_hidden_units,
            output_activation=torch.nn.Softmax(dim=1),
            random_state = self._random_state
        ).to(self._device)
        self._actor_optimizer = actor_optimizer_fn(self._actor_local.parameters())
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "alpha": prio_alpha,
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
        if not self._temperature_decay_schedule:
            self._target_entropy = 0.98 * -np.log(1 / self._action_size )
            self._log_temperature = torch.tensor(np.log(self._temperature_initial), requires_grad=True, device=self._device)
            self._temperature = self._log_temperature
            self._temperature_optimizer = critic_optimizer_fn([self._log_temperature])
        else:
            self._temperature = self._temperature_decay_schedule(self._number_episodes)
            
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
        sample_delta = self._random_state.random()
        self.delta_threshold = self._delta_decay_schedule(self._number_timesteps)
        delta_flag = sample_delta <= self.delta_threshold
        possible_actions = self.filter_actions(filter_information)
        if self.only_filter:
            if possible_actions:
                return self._random_state.choice(possible_actions)
            else:
                return self._env.action_space.sample()
        if evaluation_flag:
            discrete_action, filtered_flag = self.get_action_deterministically(state, possible_actions=possible_actions)
        else:
            discrete_action, filtered_flag = self.get_action_nondeterministically(state, possible_actions=possible_actions, delta_flag = delta_flag)
        return discrete_action, filtered_flag

    def get_action_nondeterministically(self, state, possible_actions, delta_flag):
        filtered_flag = False
        if possible_actions:
            possible_actions_tensor = torch.tensor(possible_actions, device=self._device, dtype=torch.int)
        else:
            print("No possible actions.")
        action_probabilities = self.get_action_probabilities(state)
        if delta_flag and possible_actions:
                mask = torch.ones_like(action_probabilities, dtype=bool)
                mask[possible_actions_tensor] = False
                action_probabilities[mask] = 0.0
                filtered_flag = True
        sum_action_probabilities = action_probabilities.sum()
        if sum_action_probabilities == 0.0:
            warnings.warn("Sum action probabilities is zero.")
            discrete_action = self._env.action_space.sample()
        else:
            normalized_action_probabilities = action_probabilities / sum_action_probabilities
            discrete_action = self._random_state.choice(range(self._action_size), p=normalized_action_probabilities.cpu())
        return discrete_action, filtered_flag

    def get_action_deterministically(self, state, possible_actions):
        filtered_flag = False
        if possible_actions:
            possible_actions_tensor = torch.tensor(possible_actions, device=self._device, dtype=torch.int)
        else:
            print("No possible actions.")
        action_probabilities = self.get_action_probabilities(state)
        if possible_actions:
                mask = torch.ones_like(action_probabilities, dtype=bool)
                mask[possible_actions_tensor] = False
                action_probabilities[mask] = 0.0
                filtered_flag = True
        discrete_action = torch.argmax(action_probabilities).item()
        return discrete_action, filtered_flag





    def learn(self, idxs: np.array, experiences: np.array, sampling_weights: np.array, buffer_type = "real"):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones, next_filter_masks = (torch.stack(vs,0).squeeze(1).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = actions.long()
        # dones = dones.unsqueeze(dim=1)
        
        # updates the parameters of the online network
        # Set all the gradients stored in the optimisers to zero.
        self._critic_optimizer.zero_grad()
        self._critic_optimizer2.zero_grad()
        self._actor_optimizer.zero_grad()
        if not self._temperature_decay_schedule:
            self._temperature_optimizer.zero_grad()



        if buffer_type == "real":
            _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self._device)
            deltas, critic_loss, critic2_loss = self.critic_loss(states, actions, rewards, next_states, dones, _sampling_weights, next_filter_masks)
            actor_loss, log_action_probabilities = self.actor_loss(states)
            # update experience priorities
            priorities = (deltas.abs()
                                .cpu()
                                .detach()
                                .numpy()
                                .flatten())
            self._memory.update_priorities(idxs, priorities + 1e-6) # priorities must be positive!

        elif buffer_type == "neighbour":
            _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self._device)
            deltas, critic_loss, critic2_loss = self.critic_loss(states, actions, rewards, next_states, dones, _sampling_weights, next_filter_masks, buffer_type = "neighbour")
            actor_loss, log_action_probabilities = self.actor_loss(states, sampling_weights=_sampling_weights, buffer_type = "neighbour")
        elif buffer_type == "model":
            deltas, critic_loss, critic2_loss = self.critic_loss(states, actions, rewards, next_states, dones, None, next_filter_masks)
            actor_loss, log_action_probabilities = self.actor_loss(states)
        else:
            raise ValueError("buffer_type must be 'real', 'neighbour' or 'model'")
        # compute the mean squared loss
        critic_loss.backward()
        critic2_loss.backward()
        self._critic_optimizer.step()
        self._critic_optimizer2.step()

        actor_loss.backward()
        self._actor_optimizer.step()
        if not self._temperature_decay_schedule:
            temperature_loss = self.temperature_loss(log_action_probabilities)
            temperature_loss.backward()
            self._temperature_optimizer.step()
            self._temperature = self._log_temperature.exp()
        else:
            self._temperature = self._temperature_decay_schedule(self._number_episodes)

        self.soft_update_target_networks()
        if not self._temperature_decay_schedule:
            return critic_loss, critic2_loss, actor_loss, temperature_loss
        else:
            return critic_loss, critic2_loss, actor_loss, 0.0



    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor, _sampling_weights, next_filter_masks, buffer_type = "real"):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor, next_filter_masks=next_filter_masks)
            next_q_values_target = self._critic_target.forward(next_states_tensor)
            next_q_values_target2 = self._critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self._temperature * log_action_probabilities
            )).sum(dim=1).unsqueeze(-1)

            next_q_values = rewards_tensor + ~done_tensor * self._gamma*soft_state_values

        # soft_q_values = self._critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        # soft_q_values2 = self._critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values = self._critic_local(states_tensor).gather(1, actions_tensor)
        soft_q_values2 = self._critic_local2(states_tensor).gather(1, actions_tensor)
        critic_square_deltas = next_q_values - soft_q_values
        critic2_square_deltas = next_q_values - soft_q_values2
        # deltas = [min(l1.item().abs(), l2.item().abs()) for l1, l2 in zip(critic_square_deltas, critic2_square_deltas)]
        # rewrite "deltas =" to be solvable with torch.min()
        deltas = torch.minimum(critic_square_deltas**2, critic2_square_deltas**2)
        if buffer_type == "neighbour":
            critic_loss = torch.mean((critic_square_deltas * _sampling_weights)**2)
            critic2_loss = torch.mean((critic2_square_deltas * _sampling_weights)**2)
        else:
            critic_loss = torch.mean(critic_square_deltas**2)
            critic2_loss = torch.mean(critic2_square_deltas**2)
        return deltas, critic_loss, critic2_loss

    def actor_loss(self, states_tensor, sampling_weights = None, buffer_type = "real"):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        # TODO Check if this is necessary
        with torch.no_grad():
            q_values_local = self._critic_local(states_tensor)
            q_values_local2 = self._critic_local2(states_tensor)
        entropy = self._temperature * log_action_probabilities
        inside_term = entropy - torch.minimum(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1)
        if buffer_type == "neighbour":
            sampling_weights = sampling_weights.squeeze()
            assert sampling_weights.shape == policy_loss.shape, "Policy loss & Sampling weights are not the same shape."
            policy_loss = (policy_loss * sampling_weights)
        mean_policy_loss = policy_loss.mean()
        return mean_policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        temperature_loss = -(self._log_temperature * (log_action_probabilities + self._target_entropy).detach()).mean()
        return temperature_loss

    def get_action_info(self, states_tensor, next_filter_masks = None):
        action_probabilities = self._actor_local.forward(states_tensor)
        if next_filter_masks is not None:
            next_filter_masks = next_filter_masks.bool()
            action_probabilities[next_filter_masks] = 0.0
            # softmax the action probabilities
            action_probabilities = action_probabilities/ action_probabilities.sum()
            # action_probabilities = F.softmax(action_probabilities, dim=1)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        action_probabilities = self._actor_local.forward(state)
        return action_probabilities.squeeze(0).detach().double()

    def soft_update_target_networks(self, tau=None):
        if tau:
            self.soft_update(self._critic_target, self._critic_local, tau)
            self.soft_update(self._critic_target2, self._critic_local2, tau)
        else:
            self.soft_update(self._critic_target, self._critic_local, self._default_tau)
            self.soft_update(self._critic_target2, self._critic_local2, self._default_tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self._critic_local(state)
        q_values2 = self._critic_local2(state)
        return torch.min(q_values, q_values2)
    
    def step(self, state, action, reward, next_state, done, next_filter_information,env):
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
            next_filter_mask[next_possible_actions] = 0.0 # TODO Check if correct
        next_filter_mask = next_filter_mask.to(self._device)
        experience = Experience_Symbolic(state, action.view(1,1), reward.view(1,1), next_state, torch.tensor([done]).view(1,1), next_filter_mask)
        self._memory.add(experience)
        if len(self._memory)>=self._memory.batch_size:
            self.beta = self._beta_annealing_schedule(self._number_episodes)
            idxs, experiences, sampling_weights = self._memory.sample(self.beta)
            critic_loss, critic2_loss, actor_loss, temperature_loss = self.learn(idxs, experiences, sampling_weights)
            return critic_loss, critic2_loss, actor_loss, temperature_loss
        return 0.0, 0.0, 0.0, 0.0

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
        states (np.array): the previous states of the environment.
        actions (int): the actions taken by the agent in the previous state.
        rewards (float): the rewards received from the environment.
        next_states (np.array): the resulting states of the environment following the action.
        dones (bool): True is the training episode is finised; false otherwise.
        next_filter_informations (list): the filter informations of the next states.
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

    def learn_from_buffer(self):
        """Update the agent's state based on a collection of recent simulated experiences."""
        if len(self._model_memory)>=self._model_memory.batch_size:
            self.beta = self._beta_annealing_schedule(self._number_episodes)
            idxs, experiences = self._model_memory.uniform_sample(replace = True)
            critic_loss, critic2_loss, actor_loss, temperature_loss = self.learn(idxs, experiences, None, buffer_type = "model")
            return critic_loss, critic2_loss, actor_loss, temperature_loss
        return 0.0, 0.0, 0.0, 0.0

    def add_neighbour_buffer(self, neighbour_replay_buffer) -> None:
        self._neighbour_memory = neighbour_replay_buffer

    def learn_from_neighbour_buffer(self, pretrain_flag = False):
        """Update the agent's state based on a collection of recent simulated experiences."""
        if pretrain_flag:
            num_pretraining_steps = len(self._neighbour_memory)*10//(self._neighbour_memory.batch_size)
            critic_losses = []
            critic2_losses = []
            actor_losses = []
            temperature_losses = []
            for i in range(num_pretraining_steps):
                idxs, experiences, weights = self._neighbour_memory.sample_neighbour_experience()
                critic_loss, critic2_loss, actor_loss, temperature_loss = self.learn(idxs, experiences, weights, buffer_type = "neighbour")
                critic_losses.append(critic_loss)
                critic2_losses.append(critic2_loss)
                actor_losses.append(actor_loss)
                temperature_losses.append(temperature_loss)
            return critic_losses, critic2_losses, actor_losses, temperature_losses
        if len(self._neighbour_memory)>=self._neighbour_memory.batch_size:
            idxs, experiences, weights = self._neighbour_memory.sample_neighbour_experience()
            critic_loss, critic2_loss, actor_loss, temperature_loss = self.learn(idxs, experiences, weights, buffer_type = "neighbour")
            return critic_loss, critic2_loss, actor_loss, temperature_loss
        else:
            warnings.warn("Neighbour memory is not large enough.")
        return 0.0, 0.0, 0.0, 0.0