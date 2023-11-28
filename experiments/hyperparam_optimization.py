import typing
from simulation_env.environment_basic.environment_basic import CropRotationEnv as CropRotationEnv_Basic
from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv as CropRotationEnv_Advanced
from models.basic.DQN_Prioritized import DeepQAgent as DQN_Prioritized
from models.basic.DQN_Prioritized_Symbolic import DeepQAgent as DQN_Prioritized_Symbolic
from models.advanced.SAC import SACAgent
from models.advanced.SAC_Symbolic import SACAgent as SACAgent_Symbolic
from models.advanced.fake_env import FakeEnv
import numpy as np
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
from models.utilities.ReplayBufferPrioritized import UniformReplayBuffer, Experience, Experience_Symbolic
from models.advanced.model_utilities import format_samples_for_training, create_full_replay_buffer, plot_mse_and_kl_losses_per_key, get_filter_informations_from_normalized_states, create_neighbour_replay_buffer
from tqdm import tqdm
import copy


# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

def delta_decay_schedule(steps_done, DELTA_START,DELTA_END,DELTA_DECAY):
    return DELTA_END + (DELTA_START - DELTA_END) * np.exp(-1. * steps_done / DELTA_DECAY)

def objective(trial,
        param_dict,
        agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type: str, #["basic","advanced"]
        num_episodes = 500,
        training_eval_ratio = 5,
        # Environment parameters
        DryWetInit = None,
        GroundTypeInit = None,
        deterministic = None,
        seq_len = 10,
        # Symbolic parameters
        rule_options = None,
        only_filter = False,
        # MBRL parameters
        mbrl_flag = False,
        pretrained_fake_env = None,
        neighbour_flag = False,
        neighbourhood_replay_buffer = None,
        seed = 43):
    
    random_state1 = np.random.RandomState(seed)
    random_state2 = np.random.RandomState(seed+1)    

    
    symbolic_flag = "symbolic" in agent_type
    sac_flag = "sac" in agent_type
    basic_flag = environment_type == "basic"
    assert not (basic_flag and mbrl_flag), "MBRL is not supported for the basic environment."
    weight_decay = trial.suggest_float("weight_decay",1e-9,1e-1,log=True) if "weight_decay" not in param_dict else param_dict["weight_decay"]
    batch_size = trial.suggest_int("batch_size", 32, 1024, log=True) if "batch_size" not in param_dict else param_dict["batch_size"]
    buffer_size = trial.suggest_int("buffer_size", 1000, 50000, log=True) if "buffer_size" not in param_dict else param_dict["buffer_size"]
    number_hidden_units = trial.suggest_int("number_hidden_units", 128, 1024, log=True) if "number_hidden_units" not in param_dict else param_dict["number_hidden_units"]
    beta = trial.suggest_float("beta",1e-3,1e-1, log=True) if "beta" not in param_dict else param_dict["beta"]
    tau = trial.suggest_float("tau",0.01,0.5) if "tau" not in param_dict else param_dict["tau"]
    if sac_flag:
        critic_lr = trial.suggest_float("critic_lr",1e-8,1e-2,log=True) if "critic_lr" not in param_dict else param_dict["critic_lr"]
        actor_lr = trial.suggest_float("actor_lr",1e-8,1e-2,log=True) if "actor_lr" not in param_dict else param_dict["actor_lr"]
        prio_alpha = trial.suggest_float("prio_alpha",0.1,0.9) if "prio_alpha" not in param_dict else param_dict["prio_alpha"]
        temperature_initial = trial.suggest_float("temperature_initial",0.1,3.0) if "temperature_initial" not in param_dict else param_dict["temperature_initial"]
    else:
        lr = trial.suggest_float("lr",1e-11,1e-2,log=True) if "lr" not in param_dict else param_dict["lr"]
        alpha = trial.suggest_float("alpha",0.1,0.9) if "alpha" not in param_dict else param_dict["alpha"]
        epsilon_max = trial.suggest_float("epsilon_max",0.5,0.9) if "epsilon_max" not in param_dict else param_dict["epsilon_max"]
    if symbolic_flag:
        delta_max = trial.suggest_float("delta_max",0.2,0.9) if "delta_max" not in param_dict else param_dict["delta_max"]
    if mbrl_flag:
        num_rollouts = trial.suggest_int("num_rollouts", 1, 20) if "num_rollouts" not in param_dict else param_dict["num_rollouts"]
        rollout_length = trial.suggest_int("rollout_length", 1, 5) if "rollout_length" not in param_dict else param_dict["rollout_length"]
        agent_training_steps = trial.suggest_int("agent_training_steps", 1, 20) if "agent_training_steps" not in param_dict else param_dict["agent_training_steps"]
        num_dynamics_model_training_steps = trial.suggest_int("num_dynamics_model_training_steps", 1, 50) if "num_dynamics_model_training_steps" not in param_dict else param_dict["num_dynamics_model_training_steps"]
        dynamics_model_batch_size = trial.suggest_int("dynamics_model_batch_size", 32, 1024, log=True) if "dynamics_model_batch_size" not in param_dict else param_dict["dynamics_model_batch_size"]
    if neighbour_flag:
        neighbour_alpha = trial.suggest_float("neighbour_alpha",0.1,1.0) if "neighbour_alpha" not in param_dict else param_dict["neighbour_alpha"]
    if environment_type == "basic" and symbolic_flag:
        raise ValueError("Symbolic agents can only be used with the advanced environment.")
    if symbolic_flag and not rule_options:
        raise ValueError("Symbolic agents require rule_options to be set.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set Environment
    if environment_type == "basic":
        env = CropRotationEnv_Basic(seq_len=seq_len, random_state = random_state1)
    elif environment_type == "advanced":
        env = CropRotationEnv_Advanced(
            seq_len=seq_len, 
            random_state = random_state1, 
            DryWetInit=DryWetInit, 
            GroundTypeInit=GroundTypeInit, 
            deterministic=deterministic)
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
                model_buffer_flag = mbrl_flag,
                random_state = random_state1,
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
                    random_state = random_state1,
                    model_buffer_flag = mbrl_flag,
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
                random_state = random_state1,
                rule_options = rule_options,
                only_filter = only_filter,
                model_buffer_flag = mbrl_flag,
                )
    elif agent_type == "sac_symbolic":
        agent = SACAgent_Symbolic(env = env,
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
                    random_state = random_state1,
                    rule_options = rule_options,
                    only_filter = only_filter,
                    model_buffer_flag = mbrl_flag,
                    )
    
    # MBRL Initialization
    if mbrl_flag:
        if pretrained_fake_env is not None:
            fake_env = copy.deepcopy(pretrained_fake_env)
        else:
            fake_env = FakeEnv(device = device, random_state = random_state1)
        num_dynamics_models = fake_env.get_num_models()
        dynamics_model_keys = fake_env.get_model_keys()

        experience_replay_buffer = UniformReplayBuffer(batch_size = dynamics_model_batch_size,
                                        buffer_size = 100000,
                                        random_state=random_state1)
        test_set_size = 100
        if symbolic_flag:
            test_replay_buffer = create_full_replay_buffer(length = test_set_size, seq_len=10, random_state = random_state2, DryWetInit = DryWetInit, GroundTypeInit = GroundTypeInit, batch_size = 128, device = device, filter_flag = True, rule_options = rule_options)
        else:
            test_replay_buffer = create_full_replay_buffer(length = test_set_size, seq_len=10, random_state = random_state2, DryWetInit = DryWetInit, GroundTypeInit = GroundTypeInit, batch_size = 128, device = device)
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
    if neighbour_flag:
        neighbour_buffer_copy = copy.deepcopy(neighbourhood_replay_buffer)
        neighbour_buffer_copy._neighbour_alpha = neighbour_alpha
        neighbour_buffer_copy._batch_size = batch_size
        agent.add_neighbour_buffer(neighbour_buffer_copy)
    average_last_30_rewards = 0.0
    sum_total_rewards = 0.0
    if sac_flag:
        average_last_20_critic1_losses = 100.0
        average_last_20_actor_losses = 0.0
    else:
        average_last_20_losses = 100.0
    if mbrl_flag:
        average_last_10_dynamic_mse_losses_reward = 10.0
        average_last_10_dynamic_mse_losses_sm = 1.0

    ### Neighbour pre-training
    if neighbour_flag:
        if sac_flag:
            _, _, _, _= agent.learn_from_neighbour_buffer(pretrain_flag = True)
        else:
            _ = agent.learn_from_neighbour_buffer(pretrain_flag=True)
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
            # Action selection
            if symbolic_flag:
                action, filtered_flag = agent.select_action(state, filter_information = filter_information, evaluation_flag=evaluation_flag)
            else:
                action = agent.select_action(state, evaluation_flag=evaluation_flag)

            # Environment step with action
            if not basic_flag:
                observation, next_filter_information, reward, done, info = env.step(action)
            else:
                observation, reward, done, info = env.step(action)

            reward_tensor = torch.tensor([reward], device=device)*reward_factor
            action_tensor = torch.tensor([action], device=device)

            # Set next state
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Increase episode and timestep counters for MBRL as there is no training step on actual environment
            # if mbrl_flag and not evaluation_flag:
            #     if done:
            #         agent._number_episodes += 1
            #         agent._number_timesteps += 1
            #     else:
            #         agent._number_timesteps += 1
            # Training step for agent from real environment
            if not evaluation_flag:
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


            ### Neighbour learning
            if not evaluation_flag and neighbour_flag:
                if sac_flag:
                    critic_loss, critic2_loss, actor_loss, temperature_loss = agent.learn_from_neighbour_buffer()
                    critic1_losses.append(critic_loss)
                    critic2_losses.append(critic2_loss)
                    actor_losses.append(actor_loss)
                    temperature_losses.append(temperature_loss)
                else:
                    avg_loss = agent.learn_from_neighbour_buffer()
                    losses.append(avg_loss)

            
            # Dynamics model training or evaluation
            if mbrl_flag and not evaluation_flag:
                # MBRL: Add to experience replay buffer
                if not done:
                    experience = Experience(state, action_tensor.view(1,1), reward_tensor.view(1,1), next_state, torch.tensor([done]).view(1,1))
                    experience_replay_buffer.add(experience)
                if experience_replay_buffer.buffer_length >= dynamics_model_batch_size:
                    # Dynamics model training
                    fake_env.train(
                        replay_buffer = experience_replay_buffer, 
                        num_steps = num_dynamics_model_training_steps, 
                        batch_size = dynamics_model_batch_size
                    )


            # Agent Training
            if mbrl_flag and not evaluation_flag:
                # Dynamics model rollout
                #################################
                # Sample from experience
                idxs, dynamics_experiences = experience_replay_buffer.uniform_sample(batch_size = num_rollouts, replace = True)

                dynamics_states, _, _, _, _ = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*dynamics_experiences))
                if symbolic_flag:
                    dynamics_filter_informations = get_filter_informations_from_normalized_states(dynamics_states, env)
                for rollout_step in range(rollout_length):
                    # select actions for each rollout episode
                    dynamics_action_tensors = torch.tensor([]).to(device)
                    for i_rollout_episode in range(num_rollouts):
                        if symbolic_flag:
                            dynamics_action, filtered_flag = agent.select_action(dynamics_states[i_rollout_episode:i_rollout_episode+1], filter_information = dynamics_filter_informations[i_rollout_episode], evaluation_flag=False)
                        else:
                            dynamics_action = agent.select_action(dynamics_states[i_rollout_episode:i_rollout_episode+1], evaluation_flag=False)
                        dynamics_action_tensor = torch.tensor([dynamics_action]).to(device)
                        dynamics_action_tensors = torch.cat((dynamics_action_tensors, dynamics_action_tensor), 0)
                    dynamics_action_tensors = dynamics_action_tensors.long()
                    # predict next state, reward and filter information
                    model_idxs_all_rollouts = []
                    for i_rollout_episode in range(num_rollouts):
                        model_idxs = []
                        for idx, model_key in enumerate(dynamics_model_keys):
                            model_idx = random_state1.randint(0, num_dynamics_models[idx])
                            model_idxs.append(model_idx)
                        model_idxs_all_rollouts.append(model_idxs)
                    dynamics_next_states, dynamics_rewards = fake_env.predict_batch(
                        states = dynamics_states, 
                        actions = dynamics_action_tensors,
                        model_idxs_all_rollouts = model_idxs_all_rollouts,
                        device = device
                        )
                    dynamics_dones = torch.tensor([False for i in range(num_rollouts)]).to(device).reshape(num_rollouts,1)
                    if symbolic_flag:
                        next_dynamics_filter_informations = get_filter_informations_from_normalized_states(dynamics_next_states, env)
                        agent.add_to_model_replay_buffer(dynamics_states, dynamics_action_tensors.reshape(num_rollouts,1), dynamics_rewards.reshape(num_rollouts,1), dynamics_next_states, dynamics_dones, next_dynamics_filter_informations, env)
                    else:
                        agent.add_to_model_replay_buffer(dynamics_states, dynamics_action_tensors.reshape(num_rollouts,1), dynamics_rewards.reshape(num_rollouts,1), dynamics_next_states, dynamics_dones)
                    # go to next state
                    dynamics_states = dynamics_next_states
                    if symbolic_flag:
                        dynamics_filter_informations = next_dynamics_filter_informations
                
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

        # Print results in console
        if not evaluation_flag:
            if sac_flag:
                if symbolic_flag:
                    print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
                else:
                    print(f"#{i_episode}, Reward: {total_reward}, Temperature: {agent._temperature}, Beta: {agent.beta},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
                average_last_20_critic1_losses = (average_last_20_critic1_losses * 19 + torch.tensor(critic1_losses).mean()) / 20
                average_last_20_actor_losses = (average_last_20_actor_losses * 19 + torch.tensor(actor_losses).mean()) / 20
            else:
                if symbolic_flag:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta}, Delta: {agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
                else:
                    print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {agent.eps_threshold}, Beta: {agent.beta},  Average loss: {torch.tensor(losses).mean()}")
                average_last_20_losses = (average_last_20_losses * 19 + torch.tensor(losses).mean()) / 20
            average_last_30_rewards = (average_last_30_rewards * 29 + total_reward) / 30
            sum_total_rewards = sum_total_rewards + total_reward
        else:
            print(f"#{i_episode}, Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            if mbrl_flag:
                dynamics_mean_mse_losses, dynamics_mean_kl_losses = fake_env.eval(test_inputs, test_outputs)
                for model_key in dynamics_model_keys:
                    dynamics_all_mean_mse_losses[model_key].append(np.log(dynamics_mean_mse_losses[model_key]))
                    dynamics_all_mean_kl_losses[model_key].append(np.log(dynamics_mean_kl_losses[model_key]))
                print(f"#{i_episode}, Dynamics model losses: {np.mean(dynamics_mean_mse_losses['reward'])}, {np.mean(dynamics_mean_mse_losses['stochastic_multi'])}. Dynamics model KL losses: {np.mean(dynamics_mean_kl_losses['reward'])}, {np.mean(dynamics_mean_kl_losses['stochastic_multi'])}.")
                average_last_10_dynamic_mse_losses_reward = (average_last_10_dynamic_mse_losses_reward * 9 + np.mean(dynamics_mean_mse_losses['reward'])) / 10
                average_last_10_dynamic_mse_losses_sm = (average_last_10_dynamic_mse_losses_sm * 9 + np.mean(dynamics_mean_mse_losses['stochastic_multi'])) / 10
    
    if sac_flag:
            trial.set_user_attr("avg_last_20_critic1_losses",average_last_20_critic1_losses.item())
            trial.set_user_attr("avg_last_20_actor_losses",average_last_20_actor_losses.item())    
    else:
        trial.set_user_attr("avg_last_20_losses",average_last_20_losses.item())
    if mbrl_flag:
        trial.set_user_attr("avg_last_10_dynamic_mse_losses_reward",average_last_10_dynamic_mse_losses_reward.item())
        trial.set_user_attr("avg_last_10_dynamic_mse_losses_sm",average_last_10_dynamic_mse_losses_sm.item())
    return average_last_30_rewards, sum_total_rewards/num_episodes
    

def run_optuna_study(
        agent_type: str, #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type: str, #["basic","advanced"]
        mbrl_flag = False,
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
        seed = 43,
        pretrain_buffer_size = 50000,
        pretrain_num_steps = 1000,
        neighbour_flag = False,
        num_neighbours = 20,
        neighbour_buffer_size = 5000,

        param_dict = {}
        ):
    symbolic_flag = "symbolic" in agent_type
    sac_flag = "sac" in agent_type
    basic_flag = environment_type == "basic"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state1 = np.random.RandomState(seed)
    study = optuna.create_study(directions=["maximize","maximize"])
    
    
    if mbrl_flag:
        fake_env = FakeEnv(device = device, random_state = random_state1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random_state1 = np.random.RandomState(seed)
        print("Pretrain dynamics model.")
        if symbolic_flag:
            pretrain_replay_buffer = create_full_replay_buffer(length = pretrain_buffer_size, seq_len=seq_len, random_state = random_state1, DryWetInit = DryWetInit, GroundTypeInit = GroundTypeInit, batch_size = 128, device = device, filter_flag = True, rule_options = rule_options)
        else:
            pretrain_replay_buffer = create_full_replay_buffer(length = pretrain_buffer_size, seq_len=seq_len, random_state = random_state1, DryWetInit = DryWetInit, GroundTypeInit = GroundTypeInit, batch_size = 128, device = device)
        for i in range(pretrain_num_steps):
            if i % 100 == 0:
                print(f"Pretrain-Episode {i}/{pretrain_num_steps}")
            fake_env.train(
                replay_buffer = pretrain_replay_buffer, 
                num_steps = 10, 
                batch_size = 256,
            )
        print("Pretraining of dynamics model finished.")
    else:
        fake_env = None
    if neighbour_flag:
        print("Neighbour experiment")
        env = CropRotationEnv_Advanced(seq_len=10, random_state = random_state1)
        if symbolic_flag:
            neighbour_replay_buffer = create_neighbour_replay_buffer(
                env = env, 
                num_neighbours = num_neighbours, 
                length = neighbour_buffer_size, 
                seq_len = seq_len, 
                random_state = random_state1,
                batch_size = 256,
                device = device, 
                neighbour_alpha = 1.0,
                filter_flag = True, 
                rule_options = rule_options)
        else:
            neighbour_replay_buffer = create_neighbour_replay_buffer(
                env = env, 
                num_neighbours = num_neighbours, 
                length = neighbour_buffer_size, 
                seq_len = seq_len, 
                random_state = random_state1,
                batch_size = 256,
                device = device, 
                neighbour_alpha = 1.0,
                filter_flag = False, 
                rule_options = rule_options)
    else:
        neighbour_replay_buffer = None
    # create partial function from objective function including num_episodes
    objective_partial = lambda trial: objective(
                                        trial,
                                        param_dict = param_dict,
                                        agent_type=agent_type,
                                        environment_type=environment_type,
                                        num_episodes=num_episodes,
                                        training_eval_ratio=training_eval_ratio,
                                        # Environment parameters
                                        DryWetInit=DryWetInit, 
                                        GroundTypeInit=GroundTypeInit, 
                                        deterministic=deterministic,
                                        seq_len=seq_len, 
                                        # Symbolic parameters 
                                        rule_options=rule_options, 
                                        only_filter=only_filter,
                                        # MBRL parameters
                                        mbrl_flag = mbrl_flag,
                                        pretrained_fake_env = fake_env,
                                        # Neighbour_parameters
                                        neighbour_flag = neighbour_flag,
                                        neighbourhood_replay_buffer = neighbour_replay_buffer,
                                        seed=seed)
    study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    trial_with_highest_avg_last30_rewards = max(study.best_trials, key=lambda t: t.values[0])
    trial_with_highest_episodic_rewards = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest avg rewards: ")
    print("################################")
    print(f"\tnumber: {trial_with_highest_avg_last30_rewards.number}")
    print(f"\tvalues: {trial_with_highest_avg_last30_rewards.values}")
    print(f"\tfurther attributes: {trial_with_highest_avg_last30_rewards.user_attrs}")
    print(f"\tparams:")
    pp.pprint(trial_with_highest_avg_last30_rewards.params)
    print(f"Trial with highest average episodic reward: ")
    print("################################")
    print(f"\tnumber: {trial_with_highest_episodic_rewards.number}")
    print(f"\tvalues: {trial_with_highest_episodic_rewards.values}")
    print(f"\tfurther attributes: {trial_with_highest_episodic_rewards.user_attrs}")
    print(f"\tparams:")
    pp.pprint(trial_with_highest_episodic_rewards.params)
    print("All best trials on pareto front:")
    print("################################")
    for pareto_trial in study.best_trials:
        print(f"\tTrial number: {pareto_trial.number}")
        print(f"\tvalues: {pareto_trial.values}")
        print(f"\tfurther attributes: {pareto_trial.user_attrs}")
        print(f"\tparams:")
        pp.pprint(pareto_trial.params)

