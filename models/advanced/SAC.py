import numpy as np
import torch
import typing
from models.utilities.Network import Network
from models.utilities.ReplayBufferPrioritized import PrioritizedExperienceReplayBuffer, Experience
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import random
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
                 tau:float,
                 gamma: float,
                 temperature_decay_schedule: typing.Callable[[int], float],
                 seed: int = None):
        random.seed(seed)
        self._random_state = np.random.RandomState(seed)
        self._env = env
        self._number_hidden_units = number_hidden_units
        self._gamma = gamma
        self._default_tau = tau
        self._temperature_initial = temperature_initial
        self._beta_annealing_schedule = beta_annealing_schedule
        self._temperature_decay_schedule = temperature_decay_schedule
        self.beta = None
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._critic_local = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units).to(self._device)
        self._critic_local2 = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units).to(self._device)
        
        self._critic_optimizer = critic_optimizer_fn(self._critic_local.parameters())
        self._critic_optimizer2 = critic_optimizer_fn(self._critic_local2.parameters())

        self._critic_target = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units).to(self._device)
        self._critic_target2 = Network(input_dimension=self._state_size,
                                    output_dimension=self._action_size,
                                    number_hidden_units=self._number_hidden_units).to(self._device)
        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0

        self.soft_update_target_networks(tau=1.)

        self._actor_local = Network(
            input_dimension=self._state_size,
            output_dimension=self._action_size,
            number_hidden_units=self._number_hidden_units,
            output_activation=torch.nn.Softmax(dim=1)
        ).to(self._device)
        self._actor_optimizer = actor_optimizer_fn(self._actor_local.parameters())
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "alpha": prio_alpha,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._memory = PrioritizedExperienceReplayBuffer(**_replay_buffer_kwargs)
        if not self._temperature_decay_schedule:
            self._target_entropy = 0.98 * -np.log(1 / self._action_size )
            self._log_temperature = torch.tensor(np.log(self._temperature_initial), requires_grad=True, device=self._device)
            self._temperature = self._log_temperature
            self._temperature_optimizer = critic_optimizer_fn([self._log_temperature])
        else:
            self._temperature = self._temperature_decay_schedule(self._number_episodes)

    def select_action(self, state, evaluation_flag=False):
        if evaluation_flag:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        # action_probabilities_cpu = action_probabilities.cpu().numpy().astype('float64')
        # prob_sum = action_probabilities_cpu.sum()
        normalized_action_probabilities = action_probabilities / action_probabilities.sum()
        discrete_action = random.choice(range(self._action_size), p=normalized_action_probabilities.cpu())
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = torch.argmax(action_probabilities).item()
        return discrete_action





    def learn(self, idxs: np.array, experiences: np.array, sampling_weights: np.array):
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(self._device) for vs in zip(*experiences))
        
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

        _sampling_weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self._device)
        deltas, critic_loss, critic2_loss = self.critic_loss(states, actions, rewards, next_states, dones, _sampling_weights)

        # update experience priorities
        priorities = (deltas.abs()
                            .cpu()
                            .detach()
                            .numpy()
                            .flatten())
        self._memory.update_priorities(idxs, priorities + 1e-6) # priorities must be positive!
        
        # compute the mean squared loss
        critic_loss.backward()
        critic2_loss.backward()
        self._critic_optimizer.step()
        self._critic_optimizer2.step()

        actor_loss, log_action_probabilities = self.actor_loss(states)

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



    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor, _sampling_weights):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
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
        # critic_loss = torch.mean((critic_square_deltas * _sampling_weights)**2)
        # critic2_loss = torch.mean((critic2_square_deltas * _sampling_weights)**2)
        # TODO cross check if relevant to add sampling weights again (importance sampling)
        critic_loss = torch.mean(critic_square_deltas**2)
        critic2_loss = torch.mean(critic2_square_deltas**2)
        return deltas, critic_loss, critic2_loss

    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        # TODO Check if this is necessary
        with torch.no_grad():
            q_values_local = self._critic_local(states_tensor)
            q_values_local2 = self._critic_local2(states_tensor)
        entropy = self._temperature * log_action_probabilities
        inside_term = entropy - torch.minimum(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        temperature_loss = -(self._log_temperature * (log_action_probabilities + self._target_entropy).detach()).mean()
        return temperature_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self._actor_local.forward(states_tensor)
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
    
    def step(self, state, action, reward, next_state, done):
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
            critic_loss, critic2_loss, actor_loss, temperature_loss = self.learn(idxs, experiences, sampling_weights)
            return critic_loss, critic2_loss, actor_loss, temperature_loss
        return 0.0, 0.0, 0.0, 0.0
        # transition = (state, discrete_action, reward, next_state, done)
        # self.learn(transition)