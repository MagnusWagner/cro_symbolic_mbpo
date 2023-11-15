import typing
from simulation_env.environment_basic.environment_basic import CropRotationEnv as CropRotationEnv_Basic
from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv as CropRotationEnv_Advanced
from models.basic.DQN_Prioritized import DeepQAgent as DQN_Prioritized
from models.basic.DQN_Prioritized_Symbolic import DeepQAgent as DQN_Prioritized_Symbolic
from models.advanced.SAC import SACAgent
from models.advanced.SAC_Symbolic import SACAgent as SACAgent_Symbolic
from models.advanced.fake_env import FakeEnv
import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses, plot_losses_sac
import torch
from torch import optim
from itertools import count
import collections
from numpy import random
import typing
import pprint
import optuna
from optuna.trial import TrialState
import math
from models.utilities.ReplayBufferPrioritized import UniformReplayBuffer, Experience
from models.advanced.model_utilities import format_samples_for_training, create_full_replay_buffer, plot_mse_and_kl_losses_per_key
from tqdm import tqdm


# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

def delta_decay_schedule(steps_done, DELTA_START,DELTA_END,DELTA_DECAY):
    return DELTA_END + (DELTA_START - DELTA_END) * np.exp(-1. * steps_done / DELTA_DECAY)


def single_training_run(
        param_dict,
        agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "advanced",
        num_episodes = 500,
        training_eval_ratio = 10,
        # Environment parameters
        DryWetInit = None,
        GroundTypeInit = None,
        deterministic = None,
        seq_len = 5,
        # Symbolic parameters
        only_filter = False,
        rule_options = None,
        # MBRL parameters
        mbrl_flag = False,
        seed = 43):
    
    # Set flags
    basic_flag = False
    symbolic_flag = "symbolic" in agent_type
    sac_flag = "sac" in agent_type
    if symbolic_flag and not rule_options:
        raise ValueError("Symbolic agents require rule_options to be set.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CropRotationEnv_Advanced(
        seq_len = seq_len, 
        seed = seed, 
        DryWetInit = DryWetInit, 
        GroundTypeInit = GroundTypeInit, 
        deterministic = deterministic
        )
    
    # MBRL Initialization
    if mbrl_flag:
        num_rollouts = param_dict["num_rollouts"]
        rollout_length = param_dict["rollout_length"]
        agent_training_steps = param_dict["agent_training_steps"]
        num_dynamics_model_training_steps = param_dict["num_dynamics_model_training_steps"]
        dynamics_model_batch_size = param_dict["dynamics_model_batch_size"]
        fake_env = FakeEnv(device = device, seed = seed)
        num_dynamics_models = fake_env.get_num_models()
        dynamics_model_keys = fake_env.get_model_keys()
        random_state = np.random.RandomState(seed)
        experience_replay_buffer = UniformReplayBuffer(batch_size = dynamics_model_batch_size,
                                        buffer_size = 100000,
                                        random_state=random_state)
        # TODO: Comment in for dynamics model evaluation
        test_set_size = 500
        test_replay_buffer = create_full_replay_buffer(length = test_set_size, seq_len=10, seed = seed+1, batch_size = 128, device = device)
        _, test_experiences = test_replay_buffer.uniform_sample(replace = False,batch_size=test_set_size)
        states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*test_experiences))
        test_inputs, test_outputs = format_samples_for_training(
            states=states, 
            actions = actions, 
            rewards = rewards, 
            next_states = next_states,
            device = device,
            num_actions = env.num_crops,
            )

        dynamics_all_mean_mse_losses = {}
        dynamics_all_mean_kl_losses = {}
        for model_key in dynamics_model_keys:
            dynamics_all_mean_mse_losses[model_key] = []
            dynamics_all_mean_kl_losses[model_key] = []
    # Set Agent
    if agent_type == "prioritized":
        agent = DQN_Prioritized(env = env,
                number_hidden_units = param_dict["number_hidden_units"],
                optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                # optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["lr"]),
                batch_size = param_dict["batch_size"],
                buffer_size = param_dict["buffer_size"],
                alpha = param_dict["alpha"],
                beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.3, num_episodes),
                gamma = 0.99,
                tau = param_dict["tau"],
                seed= seed,
                )
    elif agent_type == "sac":
        agent = SACAgent(env = env,
                    number_hidden_units = param_dict["number_hidden_units"],
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["critic_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
                    batch_size = param_dict["batch_size"],
                    buffer_size = param_dict["buffer_size"],
                    prio_alpha = param_dict["prio_alpha"],
                    temperature_initial = param_dict["temperature_initial"],
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                    tau = param_dict["tau"],
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["temperature_initial"], 0.01, num_episodes),
                    seed= seed,
                    )
    elif agent_type == "prioritized_symbolic":
        agent = DQN_Prioritized_Symbolic(env = env,
                number_hidden_units = param_dict["number_hidden_units"],
                optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                batch_size = param_dict["batch_size"],
                buffer_size = param_dict["buffer_size"],
                alpha = param_dict["alpha"],
                beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.01, num_episodes),
                delta_decay_schedule = lambda x: delta_decay_schedule(x, param_dict["delta_max"], 0.5, num_episodes),
                gamma = 0.99,
                tau = param_dict["tau"],
                seed= seed,
                rule_options = rule_options, # "only_break_rules_and_timing",
                only_filter = only_filter
                )
    elif agent_type == "sac_symbolic":
        agent = SACAgent_Symbolic(env = env,
                    number_hidden_units = param_dict["number_hidden_units"],
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["critic_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
                    batch_size = param_dict["batch_size"],
                    buffer_size = param_dict["buffer_size"],
                    prio_alpha = param_dict["prio_alpha"],
                    temperature_initial = param_dict["temperature_initial"],
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                    delta_decay_schedule = lambda x: delta_decay_schedule(x, param_dict["delta_max"], 0.5, num_episodes),
                    tau = param_dict["tau"],
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["temperature_initial"], 0.01, num_episodes),
                    seed= seed,
                    rule_options = rule_options, # "only_break_rules_and_timing",
                    only_filter = only_filter
                    )



    rewards = []

    # Training losses tracking
    if sac_flag:
        avg_critic1_losses = []
        avg_critic2_losses = []
        avg_actor_losses = []
        avg_temperature_losses = []
    else:
        average_losses = []
    if symbolic_flag:
        filter_dict = {
            True:np.array([]),
            False:np.array([])
        }

    for i_episode in tqdm(range(num_episodes)):
        evaluation_flag = i_episode % training_eval_ratio == 0
        total_reward = 0
        # Initialize the environment and get it's state
        if not basic_flag:
            state, filter_information = env.reset()
        else:
            state = env.reset()
        if basic_flag:
            reward_factor = 5.0/(max(env.cropYieldList.values())*1.2-env.negativeReward)
        else:
            reward_factor = 5.0/(env.max_reward-env.min_reward)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Training losses tracking
        crops_selected = []
        if sac_flag:
            critic1_losses = []
            critic2_losses = []
            actor_losses = []
            temperature_losses = []
        else:
            losses = []

        # Run episode until done
        for t in count():
            # Select action
            if symbolic_flag:
                action, filtered_flag = agent.select_action(state, filter_information = filter_information, evaluation_flag=evaluation_flag)
            else:
                action = agent.select_action(state, evaluation_flag=evaluation_flag)

            # Environment step with action
            if not basic_flag:
                observation, next_filter_information, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)
            if symbolic_flag:
                filter_dict[filtered_flag] = np.append(filter_dict[filtered_flag],reward)
            reward_tensor = torch.tensor([reward], device=device)*reward_factor
            action_tensor = torch.tensor([action]).to(device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Increase episode and timestep counters for MBRL as there is no training step on actual environment
            if mbrl_flag and not evaluation_flag:
                if done:
                    agent._number_episodes += 1
                    agent._number_timesteps += 1
                else:
                    agent._number_timesteps += 1

            # Dynamics model training or evaluation
            if mbrl_flag:
                if not evaluation_flag:
                    # MBRL: Add to experience replay buffer
                    if not done:
                        if symbolic_flag:
                            pass # TODO
                        else:
                            experience = Experience(state, action_tensor.view(1,1), reward_tensor.view(1,1), next_state, torch.tensor([done]).view(1,1))
                        experience_replay_buffer.add(experience)

                    # Dynamics model training
                    fake_env.train(
                        replay_buffer = experience_replay_buffer, 
                        num_steps = min(int(experience_replay_buffer.buffer_length/dynamics_model_batch_size)+1, num_dynamics_model_training_steps), 
                        batch_size = min(experience_replay_buffer.buffer_length, dynamics_model_batch_size)
                    )

            # Agent Training
            if mbrl_flag and not evaluation_flag:
                # Dynamics model rollout
                #################################
                # Sample from experience
                idxs, experiences = experience_replay_buffer.uniform_sample(batch_size = num_rollouts, replace = True)
                if symbolic_flag:
                    dynamics_states, _, _, _, _, filter_informations = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*experiences))
                else:
                    dynamics_states, _, _, _, _ = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*experiences))
                # for i_rollout_episode in range(num_rollouts):
                    # if symbolic_flag:
                    #     dynamics_state = dynamics_states[i_rollout_episode]
                    #     filter_information = filter_informations[i_rollout_episode]
                    # else:
                    #     dynamics_state = dynamics_states[i_rollout_episode]
                    
                    # Select action for each step in rollout and continue to next predicted state
                    # Add transitions to model replay buffer
                for rollout_step in range(rollout_length):
                    # select actions for each rollout episode
                    action_tensors = torch.tensor([]).to(device)
                    for i_rollout_episode in range(num_rollouts):
                        if symbolic_flag:
                            action, filtered_flag = agent.select_action(dynamics_states[i_rollout_episode], filter_information = filter_information, evaluation_flag=False)
                        else:
                            action = agent.select_action(dynamics_states[i_rollout_episode], evaluation_flag=False)
                        action_tensor = torch.tensor([action]).to(device)
                        action_tensors = torch.cat((action_tensors, action_tensor), 0)
                    action_tensors = action_tensors.long()
                    # predict next state, reward and filter information
                    model_idxs_all_rollouts = []
                    for i_rollout_episode in range(num_rollouts):
                        model_idxs = []
                        for idx, model_key in enumerate(dynamics_model_keys):
                            model_idx = random_state.randint(0, num_dynamics_models[idx])
                            model_idxs.append(model_idx)
                        model_idxs_all_rollouts.append(model_idxs)
                    dynamics_next_states, dynamics_rewards, filter_informations = fake_env.predict_batch(
                        states = dynamics_states, 
                        actions = action_tensors,
                        model_idxs_all_rollouts = model_idxs_all_rollouts,
                        device = device
                        )
                    dones = torch.tensor([False for i in range(num_rollouts)]).to(device).reshape(num_rollouts,1)
                    if symbolic_flag:
                        pass # TODO add everything INCLUDING filter information to model replay buffer
                    else:
                        agent.add_to_model_replay_buffer(dynamics_states, action_tensors.reshape(num_rollouts,1), dynamics_rewards.reshape(num_rollouts,1), dynamics_next_states, dones)
                    # go to next state
                    dynamics_states = dynamics_next_states
                
                # Agent training
                for i_agent_training in range(agent_training_steps):
                    if sac_flag:
                        critic_loss, critic2_loss, actor_loss, temperature_loss = agent.learn_from_buffer()
                        critic1_losses.append(critic_loss)
                        critic2_losses.append(critic2_loss)
                        actor_losses.append(actor_loss)
                        temperature_losses.append(temperature_loss)
                    else:
                        avg_loss = agent.learn_from_buffer()
                        losses.append(avg_loss)
            # Agent step without MBRL
            elif not mbrl_flag and not evaluation_flag:
                if sac_flag:
                    if symbolic_flag:
                        critic_loss, critic2_loss, actor_loss, temperature_loss = agent.step(state, action, reward_tensor, next_state, done, next_filter_information, env)
                    else:
                        critic_loss, critic2_loss, actor_loss, temperature_loss = agent.step(state, action, reward_tensor, next_state, done)
                    critic1_losses.append(critic_loss)
                    critic2_losses.append(critic2_loss)
                    actor_losses.append(actor_loss)
                    temperature_losses.append(temperature_loss)
                else:
                    if symbolic_flag:
                        avg_loss = agent.step(state, action, reward_tensor, next_state, done, next_filter_information, env)
                    else:
                        avg_loss = agent.step(state, action, reward_tensor, next_state, done)
                    losses.append(avg_loss)
            # Move to the next state
            state = next_state
            total_reward += reward
            if not basic_flag:
                filter_information = next_filter_information
            if info["Previous crop"] in ["CLOVER GRASS","ALFALFA"]: 
                crops_selected.append(info["Previous crop"])
            crops_selected.append(info["Previous crop"])
            if done:
                break

        # Print results in console
        if not evaluation_flag:
            if sac_flag:
                if symbolic_flag:
                    print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
                else:
                    print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
                # print(f"Selected crops: {crops_selected}")
                avg_critic1_losses.append(torch.tensor(critic1_losses).mean())
                avg_critic2_losses.append(torch.tensor(critic2_losses).mean())
                avg_actor_losses.append(torch.tensor(actor_losses).mean())
                avg_temperature_losses.append(torch.tensor(temperature_losses).mean())
            else:
                if symbolic_flag:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
                else:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta},  Average loss: {torch.tensor(losses).mean()}")
                average_losses.append(torch.tensor(losses).mean())
        else:                
            print(f"#{i_episode}, Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            if mbrl_flag:
                dynamics_mean_mse_losses, dynamics_mean_kl_losses = fake_env.eval(test_inputs, test_outputs)
                for model_key in dynamics_model_keys:
                    dynamics_all_mean_mse_losses[model_key].append(np.log(dynamics_mean_mse_losses[model_key]))
                    dynamics_all_mean_kl_losses[model_key].append(np.log(dynamics_mean_kl_losses[model_key]))
                print(f"#{i_episode}, Dynamics model losses: {np.mean(dynamics_mean_mse_losses['reward'])}, {np.mean(dynamics_mean_mse_losses['stochastic_multi'])}. Dynamics model KL losses: {np.mean(dynamics_mean_kl_losses['reward'])}, {np.mean(dynamics_mean_kl_losses['stochastic_multi'])}.")
            rewards.append(total_reward)
            if symbolic_flag:
                _ = 1 # TODO Change to show filtered difference to non-filtered
                # print(f"Filtered difference: {filter_dict[True].mean() - filter_dict[False].mean()}")
    print('Complete')
    plot_experiment(rewards)
    if sac_flag:
        plot_losses_sac(avg_critic1_losses, avg_critic2_losses, avg_actor_losses)
        results = {
            "avg_critic1_losses":avg_critic1_losses,
            "avg_critic2_losses":avg_critic2_losses,
            "avg_actor_losses":avg_actor_losses,
            "avg_temperature_losses":avg_temperature_losses,
            "rewards":rewards
        }
    else:
        plot_losses(np.log(average_losses))
        results = {
            "average_losses":average_losses,
            "rewards":rewards
        }
    if mbrl_flag:
        plot_mse_and_kl_losses_per_key(dynamics_all_mean_mse_losses, dynamics_all_mean_kl_losses)

    return results






# def single_training_run(
#         param_dict,
#         agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
#         environment_type: str, #["basic","advanced"]
#         rule_options = None,
#         num_episodes = 500,
#         training_eval_ratio = 5,
#         DryWetInit = None,
#         GroundTypeInit = None,
#         deterministic = None,
#         only_filter = False,
#         seq_len = 5,
#         seed = 43):
#     symbolic_flag = "symbolic" in agent_type
#     sac_flag = "sac" in agent_type
#     basic_flag = environment_type == "basic"
#     if environment_type == "basic" and symbolic_flag:
#         raise ValueError("Symbolic agents can only be used with the advanced environment.")
#     if symbolic_flag and not rule_options:
#         raise ValueError("Symbolic agents require rule_options to be set.")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Set Environment
#     if environment_type == "basic":
#         env = CropRotationEnv_Basic(seq_len=seq_len, seed = seed)
#     elif environment_type == "advanced":
#         env = CropRotationEnv_Advanced(seq_len=seq_len, seed = seed, DryWetInit=DryWetInit, GroundTypeInit=GroundTypeInit, deterministic=deterministic)
    
#     # Set Agent
#     if agent_type == "prioritized":
#         agent = DQN_Prioritized(env = env,
#                 number_hidden_units = param_dict["number_hidden_units"],
#                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                 # optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["lr"]),
#                 batch_size = param_dict["batch_size"],
#                 buffer_size = param_dict["buffer_size"],
#                 alpha = param_dict["alpha"],
#                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
#                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.01, num_episodes),
#                 gamma = 0.99,
#                 tau = param_dict["tau"],
#                 seed= seed,
#                 )
#     elif agent_type == "sac":
#         agent = SACAgent(env = env,
#                     number_hidden_units = param_dict["number_hidden_units"],
#                     critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["critic_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                     actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                     # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
#                     # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
#                     batch_size = param_dict["batch_size"],
#                     buffer_size = param_dict["buffer_size"],
#                     prio_alpha = param_dict["prio_alpha"],
#                     temperature_initial = param_dict["temperature_initial"],
#                     beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
#                     tau = param_dict["tau"],
#                     gamma = 0.99,
#                     temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["temperature_initial"], 0.01, num_episodes),
#                     seed= seed,
#                     )
#     elif agent_type == "prioritized_symbolic":
#         agent = DQN_Prioritized_Symbolic(env = env,
#                 number_hidden_units = param_dict["number_hidden_units"],
#                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                 batch_size = param_dict["batch_size"],
#                 buffer_size = param_dict["buffer_size"],
#                 alpha = param_dict["alpha"],
#                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
#                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.01, num_episodes),
#                 delta_decay_schedule = lambda x: delta_decay_schedule(x, param_dict["delta_max"], 0.5, num_episodes),
#                 gamma = 0.99,
#                 tau = param_dict["tau"],
#                 seed= seed,
#                 rule_options = rule_options, # "only_break_rules_and_timing",
#                 only_filter = only_filter
#                 )
#     elif agent_type == "sac_symbolic":
#         agent = SACAgent_Symbolic(env = env,
#                     number_hidden_units = param_dict["number_hidden_units"],
#                     critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["critic_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                     actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
#                     # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
#                     # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
#                     batch_size = param_dict["batch_size"],
#                     buffer_size = param_dict["buffer_size"],
#                     prio_alpha = param_dict["prio_alpha"],
#                     temperature_initial = param_dict["temperature_initial"],
#                     beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
#                     delta_decay_schedule = lambda x: delta_decay_schedule(x, param_dict["delta_max"], 0.5, num_episodes),
#                     tau = param_dict["tau"],
#                     gamma = 0.99,
#                     temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["temperature_initial"], 0.01, num_episodes),
#                     seed= seed,
#                     rule_options = rule_options, # "only_break_rules_and_timing",
#                     only_filter = only_filter
#                     )

#     rewards = []

#     if sac_flag:
#         avg_critic1_losses = []
#         avg_critic2_losses = []
#         avg_actor_losses = []
#         avg_temperature_losses = []
#     else:
#         average_losses = []
#     if symbolic_flag:
#         filter_dict = {
#             True:np.array([]),
#             False:np.array([])
#         }

#     for i_episode in range(num_episodes):
#         evaluation_flag = i_episode % training_eval_ratio == 0
#         total_reward = 0
#         # Initialize the environment and get it's state
#         if not basic_flag:
#             state, filter_information = env.reset()
#         else:
#             state = env.reset()
#         if basic_flag:
#             reward_factor = 5.0/(max(env.cropYieldList.values())*1.2-env.negativeReward)
#         else:
#             reward_factor = 5.0/(env.max_reward-env.min_reward)
#         state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#         crops_selected = []
#         if sac_flag:
#             critic1_losses = []
#             critic2_losses = []
#             actor_losses = []
#             temperature_losses = []
#         else:
#             losses = []

#         for t in count():
#             if symbolic_flag:
#                 action, filtered_flag = agent.select_action(state, filter_information = filter_information, evaluation_flag=evaluation_flag)
#             else:
#                 action = agent.select_action(state, evaluation_flag=evaluation_flag)
#             if not basic_flag:
#                 observation, next_filter_information, reward, done, info = env.step(action)
#             else:
#                 observation, reward, done, info = env.step(action)
#             if symbolic_flag:
#                 filter_dict[filtered_flag] = np.append(filter_dict[filtered_flag],reward)
#             reward_tensor = torch.tensor([reward], device=device)*reward_factor
#             if done:
#                 next_state = None
#             else:
#                 next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
#             if sac_flag:
#                 if not evaluation_flag:
#                     if symbolic_flag:
#                         critic_loss, critic2_loss, actor_loss, temperature_loss = agent.step(state, action, reward_tensor, next_state, done, next_filter_information, env)
#                     else:
#                         critic_loss, critic2_loss, actor_loss, temperature_loss = agent.step(state, action, reward_tensor, next_state, done)
#                     critic1_losses.append(critic_loss)
#                     critic2_losses.append(critic2_loss)
#                     actor_losses.append(actor_loss)
#                     temperature_losses.append(temperature_loss)
#             else:
#                 if symbolic_flag:
#                     avg_loss = agent.step(state, action, reward_tensor, next_state, done, next_filter_information, env)
#                 else:
#                     avg_loss = agent.step(state, action, reward_tensor, next_state, done)
#                 losses.append(avg_loss)
#             # Move to the next state
#             state = next_state
#             total_reward += reward
#             if not basic_flag:
#                 filter_information = next_filter_information
#             if info["Previous crop"] in ["CLOVER GRASS","ALFALFA"]: 
#                 crops_selected.append(info["Previous crop"])
#             crops_selected.append(info["Previous crop"])
#             if done:
#                 break
#         if not evaluation_flag:
#             if sac_flag:
#                 if symbolic_flag:
#                     print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
#                 else:
#                     print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
#                 # print(f"Selected crops: {crops_selected}")
#                 avg_critic1_losses.append(torch.tensor(critic1_losses).mean())
#                 avg_critic2_losses.append(torch.tensor(critic2_losses).mean())
#                 avg_actor_losses.append(torch.tensor(actor_losses).mean())
#                 avg_temperature_losses.append(torch.tensor(temperature_losses).mean())
#             else:
#                 if symbolic_flag:
#                     print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
#                 else:
#                     print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta},  Average loss: {torch.tensor(losses).mean()}")
#                 average_losses.append(torch.tensor(losses).mean())
#         else:
#             print(f"#{i_episode}, Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
#             rewards.append(total_reward)
#             if symbolic_flag:
#                 _ = 1 # TODO Change to show filtered difference to non-filtered
#                 # print(f"Filtered difference: {filter_dict[True].mean() - filter_dict[False].mean()}")
#     print('Complete')
#     plot_experiment(rewards)
#     if sac_flag:
#         plot_losses_sac(avg_critic1_losses, avg_critic2_losses, avg_actor_losses)
#         results = {
#             "avg_critic1_losses":avg_critic1_losses,
#             "avg_critic2_losses":avg_critic2_losses,
#             "avg_actor_losses":avg_actor_losses,
#             "avg_temperature_losses":avg_temperature_losses,
#             "rewards":rewards
#         }
#     else:
#         plot_losses(np.log(average_losses))
#         results = {
#             "average_losses":average_losses,
#             "rewards":rewards
#         }

#     return results

def objective(trial,
        agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type: str, #["basic","advanced"]
        num_episodes = 500,
        training_eval_ratio = 5,
        DryWetInit = None,
        GroundTypeInit = None,
        deterministic = None,
        rule_options = None,
        only_filter = False,
        seq_len = 5,
        seed = 43):
    symbolic_flag = "symbolic" in agent_type
    sac_flag = "sac" in agent_type
    basic_flag = environment_type == "basic"

    weight_decay = trial.suggest_float("weight_decay",1e-9,1e-1,log=True)
    batch_size = trial.suggest_int("batch_size", 32, 1024, log=True)
    buffer_size = trial.suggest_int("buffer_size", 1000, 100000, log=True)
    number_hidden_units = trial.suggest_int("number_hidden_units", 32, 1024, log=True)
    beta = trial.suggest_float("beta",1e-3,1e-2, log=True)
    tau = trial.suggest_float("tau",0.01,0.5)
    if sac_flag:
        critic_lr = trial.suggest_float("critic_lr",1e-8,1e-2,log=True)
        actor_lr = trial.suggest_float("actor_lr",1e-8,1e-2,log=True)
        prio_alpha = trial.suggest_float("prio_alpha",0.1,0.9)
        temperature_initial = trial.suggest_float("temperature_initial",0.1,3.0)
    else:
        lr = trial.suggest_float("lr",1e-8,1e-2,log=True)
        alpha = trial.suggest_float("alpha",0.1,0.9)
        epsilon_max = trial.suggest_float("epsilon_max",0.2,0.9)
    if symbolic_flag:
        delta_max = trial.suggest_float("delta_max",0.2,0.9)

    if environment_type == "basic" and symbolic_flag:
        raise ValueError("Symbolic agents can only be used with the advanced environment.")
    if symbolic_flag and not rule_options:
        raise ValueError("Symbolic agents require rule_options to be set.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set Environment
    if environment_type == "basic":
        env = CropRotationEnv_Basic(seq_len=seq_len, seed = seed)
    elif environment_type == "advanced":
        env = CropRotationEnv_Advanced(seq_len=seq_len, seed = seed, DryWetInit=DryWetInit, GroundTypeInit=GroundTypeInit, deterministic=deterministic)
    
    # Set Agent
    if agent_type == "prioritized":
        agent = DQN_Prioritized(env = env,
                number_hidden_units = number_hidden_units,
                optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=lr, amsgrad=False, weight_decay = weight_decay),
                # optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["lr"]),
                batch_size = batch_size,
                buffer_size = buffer_size,
                alpha = alpha,
                beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, epsilon_max, 0.01, num_episodes),
                gamma = 0.99,
                tau = tau,
                seed= seed,
                )
    elif agent_type == "sac":
        agent = SACAgent(env = env,
                    number_hidden_units = number_hidden_units,
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=critic_lr, amsgrad=False, weight_decay = weight_decay),
                    actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=actor_lr, amsgrad=False, weight_decay = weight_decay),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
                    batch_size = batch_size,
                    buffer_size = buffer_size,
                    prio_alpha = prio_alpha,
                    temperature_initial = temperature_initial,
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x,beta),
                    tau = tau,
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, temperature_initial, 0.01, num_episodes),
                    seed= seed,
                    )
    elif agent_type == "prioritized_symbolic":
        agent = DQN_Prioritized_Symbolic(env = env,
                number_hidden_units = number_hidden_units,
                optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=lr, amsgrad=False, weight_decay = weight_decay),
                batch_size = batch_size,
                buffer_size = buffer_size,
                alpha = alpha,
                beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, epsilon_max, 0.1, num_episodes),
                delta_decay_schedule = lambda x: delta_decay_schedule(x, delta_max, 0.3, num_episodes),
                gamma = 0.99,
                tau = tau,
                seed= seed,
                rule_options = rule_options,
                only_filter = only_filter
                )
    elif agent_type == "sac_symbolic":
        agent = SACAgent(env = env,
                    number_hidden_units = number_hidden_units,
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=critic_lr, amsgrad=False, weight_decay = weight_decay),
                    actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=actor_lr, amsgrad=False, weight_decay = weight_decay),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    # actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
                    batch_size = batch_size,
                    buffer_size = buffer_size,
                    prio_alpha = prio_alpha,
                    temperature_initial = temperature_initial,
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x,beta),
                    delta_decay_schedule = lambda x: delta_decay_schedule(x, delta_max, 0.3, num_episodes),
                    tau = tau,
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, temperature_initial, 0.01, num_episodes),
                    seed= seed,
                    rule_options = rule_options,
                    only_filter = only_filter
                    )
    average_last_10_rewards = 0.0
    if sac_flag:
        average_last_20_critic1_losses = 100.0
        average_last_20_actor_losses = 0.0
    else:
        average_last_20_losses = 100.0


    for i_episode in range(num_episodes):
        evaluation_flag = i_episode % training_eval_ratio == 0
        total_reward = 0
        # Initialize the environment and get it's state
        if not basic_flag:
            state, filter_information = env.reset()
        else:
            state = env.reset()
        if basic_flag:
            reward_factor = 5.0/(max(env.cropYieldList.values())*1.2-env.negativeReward)
        else:
            reward_factor = 5.0/(env.max_reward-env.min_reward)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        if sac_flag:
            critic1_losses = []
            critic2_losses = []
            actor_losses = []
            temperature_losses = []
        else:
            losses = []

        for t in count():
            # Action selection
            if symbolic_flag:
                action, filtered_flag = agent.select_action(state, filter_information = filter_information, evaluation_flag=evaluation_flag)
            else:
                action = agent.select_action(state, evaluation_flag=evaluation_flag)

            # Environment Step
            if not basic_flag:
                observation, next_filter_information, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)

            # Set reward for experience
            reward_tensor = torch.tensor([reward], device=device)*reward_factor

            # Set next state
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Training step for agent
            if sac_flag:
                if not evaluation_flag:
                    critic_loss, critic2_loss, actor_loss, temperature_loss = agent.step(state, action, reward_tensor, next_state, done)
                    critic1_losses.append(critic_loss)
                    critic2_losses.append(critic2_loss)
                    actor_losses.append(actor_loss)
                    temperature_losses.append(temperature_loss)
            else:
                if symbolic_flag:
                    avg_loss = agent.step(state, action, reward_tensor, next_state, done, next_filter_information, env)
                else:
                    avg_loss = agent.step(state, action, reward_tensor, next_state, done)
                losses.append(avg_loss)
            
            # Move to the next state
            state = next_state

            # Add single reward to total episodic reward
            total_reward += reward

            # Update filter information
            if not basic_flag:
                filter_information = next_filter_information

            # Add previous crop to list of crops selected
            # Add CLOVER GRASS and ALFALFA twice (due to them being on the field for two years)
            if info["Previous crop"] in ["CLOVER GRASS","ALFALFA"]: 
                crops_selected.append(info["Previous crop"])
            crops_selected.append(info["Previous crop"])

            # break if episode is done
            if done:
                break
        if not evaluation_flag:
            if sac_flag:
                print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
                average_last_20_critic1_losses = (average_last_20_critic1_losses * 19 + torch.tensor(critic1_losses).mean()) / 20
                average_last_20_actor_losses = (average_last_20_actor_losses * 19 + torch.tensor(actor_losses).mean()) / 20
            else:
                if symbolic_flag:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
                else:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta},  Average loss: {torch.tensor(losses).mean()}")
                average_last_20_losses = (average_last_20_losses * 19 + torch.tensor(losses).mean()) / 20
        else:
            print(f"#{i_episode}, Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            average_last_10_rewards = (average_last_10_rewards * 9 + total_reward) / 10
    if sac_flag:
        return average_last_20_critic1_losses, average_last_20_actor_losses, average_last_10_rewards
    else:
        return average_last_20_losses, average_last_10_rewards
    

def run_optuna_study(
        agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type: str, #["basic","advanced"]
        n_trials=30,
        timeout=600,
        num_episodes = 500,
        training_eval_ratio = 5,
        DryWetInit = None,
        GroundTypeInit = None,
        deterministic = None,
        rule_options = None,
        only_filter = False,
        seq_len = 5,
        seed = 43):
    symbolic_flag = "symbolic" in agent_type
    sac_flag = "sac" in agent_type
    basic_flag = environment_type == "basic"
    if sac_flag:
        study = optuna.create_study(directions=["minimize","minimize","maximize"])
    else:
        study = optuna.create_study(directions=["minimize","maximize"])
    # create partial function from objective function including num_episodes
    objective_partial = lambda trial: objective(
                                        trial,  
                                        agent_type=agent_type, 
                                        environment_type=environment_type,
                                        num_episodes=num_episodes,
                                        training_eval_ratio=training_eval_ratio, 
                                        DryWetInit=DryWetInit, 
                                        GroundTypeInit=GroundTypeInit, 
                                        deterministic=deterministic, 
                                        rule_options=rule_options, 
                                        only_filter=only_filter, 
                                        seq_len=seq_len, 
                                        seed=seed)
    study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    if sac_flag:
        trial_with_highest_avg_rewards = max(study.best_trials, key=lambda t: t.values[2])
    else:
        trial_with_highest_avg_rewards = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest avg rewards: ")
    print("################################")
    print(f"\tnumber: {trial_with_highest_avg_rewards.number}")
    print(f"\tvalues: {trial_with_highest_avg_rewards.values}")
    print(f"\tparams:")
    pp.pprint(trial_with_highest_avg_rewards.params)
    print("All best trials on pareto front:")
    print("################################")
    for best_trial in study.best_trials:
        print(f"\tTrial number: {best_trial.number}")
        print(f"\tvalues: {best_trial.values}")
        print(f"\tparams:")
        pp.pprint(best_trial.params)


def check_filter(agent_type: str): #["prioritized_symbolic","sac_symbolic"]
    num_episodes = 100
    seed = 3
    env = CropRotationEnv_Advanced(seq_len=10, seed = seed)
    if agent_type == "prioritized_symbolic":
        agent = DQN_Prioritized_Symbolic(env = env,
                    number_hidden_units = 1024,
                    optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=2e-6, amsgrad=False, weight_decay = 1e-2),
                    batch_size = 512,
                    buffer_size = 100000,
                    alpha = 0.4,
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, 2e-3),
                    epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, 0.5, 0.1, num_episodes),
                    delta_decay_schedule = lambda x: delta_decay_schedule(x, 0.9, 0.3, num_episodes),
                    gamma = 0.99,
                    tau = 0.1,
                    seed= seed,
                    rule_options = "humus_and_breaks",
                    only_filter = True
                    )
    else:
        pass
    n_broken_rules = []
    rewards = []
    broken_rewards = []
    count_broken_rules = 0
    no_possible_actions = 0
    for episode in range(1, num_episodes+1):
        x = random.random()
        state, filter_information = env.reset()
        done = False
        score = 0 
        while not done:
            possible_actions = agent.filter_actions(filter_information)
            if possible_actions:
                if x >= 0.5:
                    action = np.random.choice(possible_actions)
                    observation, next_filter_information, reward, done, info = env.step(action)
                    rewards.append(reward)
                    n_broken_rules.append(info["Num broken rules"])
                    if info["Num broken rules"] > 0:
                        count_broken_rules += 1
                    filter_information = next_filter_information
                else:
                    action = env.action_space.sample()
                    observation, filter_information, reward, done, info = env.step(action)
                    broken_rewards.append(reward)
            else:
                action = env.action_space.sample()
                print(f"No possible action. Taking random action: {action}.")
                no_possible_actions += 1
                observation, filter_information, reward, done, info = env.step(action)
                broken_rewards.append(reward)
    print("Number of events without possible actions:" , no_possible_actions)
    print("% of events with possible actions but broken rules:", count_broken_rules / (len(n_broken_rules) + no_possible_actions) * 100)
    # Plot the number of broken rules for each episode
    import matplotlib.pyplot as plt
    plt.plot(n_broken_rules)
    plt.title('Number of broken rules')
    plt.xlabel('Episode')
    plt.ylabel('Number of broken rules')
    plt.show()
    # Plot the rewards and the broken rewards in the same plot with a legend
    plt.plot(rewards)
    plt.plot(broken_rewards)
    plt.legend(["Rewards", "Rewards for broken rules"])
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

