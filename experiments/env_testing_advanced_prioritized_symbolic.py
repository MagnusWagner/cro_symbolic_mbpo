from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
from models.basic.DQN_Prioritized_Symbolic import DeepQAgent
import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import collections
import typing
import pprint
import optuna
from optuna.trial import TrialState
from numpy import random
import math

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def plot_prices_and_costs():
    env = CropRotationEnv(seed = 43, seq_len = 10)
    # Generate 5 random crop rotations without training (for enviornment testing)

    state, filter_information = env.reset()
    prices = env.prices.reshape((1,len(env.prices)))
    sowing_costs = env.sowing_costs.reshape((1,len(env.sowing_costs)))
    other_costs = env.other_costs.reshape((1,len(env.other_costs)))
    N_costs = np.array([env.N_costs]).reshape((1,1))
    P_costs = np.array([env.P_costs]).reshape((1,1))
    K_costs = np.array([env.K_costs]).reshape((1,1))
    done = False
    while not done:
        action = env.action_space.sample()
        observation, filter_information, reward, done, _ = env.step(action)
        prices = np.vstack((prices, env.prices.reshape((1,len(env.prices)))))
        sowing_costs = np.vstack((sowing_costs, env.sowing_costs.reshape((1,len(env.sowing_costs)))))
        other_costs = np.vstack((other_costs, env.other_costs.reshape((1,len(env.other_costs)))))
        # Calculate N_costs, P_costs and K_costs which are not arrays but floats
        N_costs_tmp = np.array([env.N_costs])
        P_costs_tmp = np.array([env.P_costs])
        K_costs_tmp = np.array([env.K_costs])
        N_costs = np.vstack((N_costs, N_costs_tmp.reshape((1,1))))
        P_costs = np.vstack((P_costs, P_costs_tmp.reshape((1,1))))
        K_costs = np.vstack((K_costs, K_costs_tmp.reshape((1,1))))
    first_prices = prices[0]
    first_sowing_costs = sowing_costs[0]
    first_other_costs = other_costs[0]
    first_N_costs = N_costs[0]
    first_P_costs = P_costs[0]
    first_K_costs = K_costs[0]
    # Calculate normalized prices and costs
    normalized_prices = (prices-first_prices) / first_prices
    normalized_sowing_costs = (sowing_costs-first_sowing_costs) / first_sowing_costs
    normalized_other_costs = (other_costs-first_other_costs) / first_other_costs
    normalized_N_costs =  (N_costs-first_N_costs) / first_N_costs
    normalized_P_costs = (P_costs-first_P_costs) / first_P_costs
    normalized_K_costs = (K_costs-first_K_costs) / first_K_costs
    # Plot each price and cost variable over time where each row (first dimension) represents a time point and each column (second dimension) represents a different crop (indices 0 to 22)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(6, 1, figsize=(10, 10))
    axs[0].plot(normalized_prices)
    axs[0].set_title('Normalized prices')
    axs[1].plot(normalized_sowing_costs)
    axs[1].set_title('Normalized sowing costs')
    axs[2].plot(normalized_other_costs)
    axs[2].set_title('Normalized other costs')
    axs[3].plot(normalized_N_costs)
    axs[3].set_title('Normalized N costs')
    axs[4].plot(normalized_P_costs)
    axs[4].set_title('Normalized P costs')
    axs[5].plot(normalized_K_costs)
    axs[5].set_title('Normalized K costs')
    for ax in axs.flat:
        ax.set(xlabel='Crop', ylabel='Normalized value')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def check_filter():
    num_episodes = 100
    seed = 3
    env = CropRotationEnv(seq_len=10, seed = seed)
    dqn_agent = DeepQAgent(env = env,
                 number_hidden_units = 1024,
                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=2e-6, amsgrad=False, weight_decay = 1e-2),
                 batch_size = 512,
                 buffer_size = 100000,
                 alpha = 0.4,
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, 2e-3),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, 0.5, 0.1, num_episodes),
                 delta_decay_schedule = lambda x: delta_decay_schedule(x, 0.9, 0.3, num_episodes),
                 gamma = 0.99,
                 update_frequency = 1,
                 seed= seed,
                 rule_options = "only_break_rules_and_timing",
                 only_filter = True
                 )
    n_broken_rules = []
    rewards = []
    broken_rewards = []
    count_broken_rules = 0
    no_possible_actions = 0
    for episode in range(1, num_episodes+1):
        state, filter_information = env.reset()
        done = False
        score = 0 
        while not done:
            possible_actions = dqn_agent.filter_actions(filter_information)
            if possible_actions:
                x = random.random()
                if x >= 0.5:
                    action = np.random.choice(possible_actions)
                    observation, filter_information, reward, done, info = env.step(action)
                    rewards.append(reward)
                    n_broken_rules.append(info["Num broken rules"])
                    if info["Num broken rules"] > 0:
                        count_broken_rules += 1
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




def constant_annealing_schedule(n, constant):
    return constant


def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

def delta_decay_schedule(steps_done, DELTA_START,DELTA_END,DELTA_DECAY):
    return DELTA_END + (DELTA_START - DELTA_END) * np.exp(-1. * steps_done / DELTA_DECAY)


def test_run():
    # max_opt_steps = 10
    param_dict = {
            "lr": 3.66e-07,
            "weight_decay": 0.00487,
            "batch_size": 310,
            "buffer_size": 251840,
            "number_hidden_units": 45,
            "alpha": 0.523,
            "beta": 0.00871,
            "epsilon_max": 0.99,
    }
    rewards = []
    average_losses = []

    seed = 41
    env = CropRotationEnv(seq_len=8, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    dqn_agent = DeepQAgent(env = env,
                 number_hidden_units = param_dict["number_hidden_units"],
                 optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["lr"]),  #, amsgrad=False, weight_decay = param_dict["weight_decay"]),
                 batch_size = param_dict["batch_size"],
                 buffer_size = param_dict["buffer_size"],
                 alpha = param_dict["alpha"],
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.3, num_episodes),
                 delta_decay_schedule = lambda x: delta_decay_schedule(x, 0.5, 0.1, num_episodes),
                 gamma = 0.99,
                 update_frequency = 1,
                 seed= seed,
                 rule_options = "humus_and_breaks", # "only_break_rules_and_timing",
                 only_filter = False
                 )
    filter_dict = {
        True:np.array([]),
        False:np.array([])
    }
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state, filter_information = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        losses = []
        for t in count():
            action, filtered_flag = dqn_agent.select_action(state, filter_information)
            observation, next_filter_information, reward, done, info = env.step(action.item())
            filter_dict[filtered_flag] = np.append(filter_dict[filtered_flag],reward)
            # observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            avg_loss = dqn_agent.step(state, action, reward, next_state, done, filter_information, env)

            # Move to the next state
            losses.append(avg_loss)
            state = next_state
            total_reward += reward.item()
            filter_information = next_filter_information
            crops_selected.append(info["Previous crop"])
            if done:
                break
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta}, Delta: {dqn_agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
        print(f"Selected crops: {crops_selected}")
        # print(f"Filtered difference: {filter_dict[True].mean() - filter_dict[False].mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
    total_reward = 0
    # Initialize the environment and get it's state
    state, filter_information = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    crops_selected = []
    total_reduction_factors = []
    losses = []
    for t in count():
        action, filtered_flag = dqn_agent.select_action(state, filter_information, greedy=True)
        observation, next_filter_information, reward, done, info = env.step(action.item())
        total_reward += reward
        filter_information = next_filter_information
        crops_selected.append(info["Previous crop"])
        total_reduction_factors.append(info["Total Reduction Factor"])
        pp.pprint(info)
        if done:
            next_state = None
            break
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    print("Total reward:",total_reward)
    print("Crops selected:", crops_selected)
    print("Total reduction factors:", total_reduction_factors)
    print('Complete')
    plot_experiment(rewards)
    plot_losses(np.log(average_losses))

def objective(trial, num_episodes = 1000):
    lr = trial.suggest_float("lr",1e-8,1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-9,1e-1,log=True)
    batch_size = trial.suggest_int("batch_size", 32, 1024, log=True)
    buffer_size = trial.suggest_int("buffer_size", 1000, 100000, log=True)
    number_hidden_units = trial.suggest_int("number_hidden_units", 32, 1024, log=True)
    rewards = []
    average_losses = []
    alpha = trial.suggest_float("alpha",0.1,0.9)
    beta = trial.suggest_float("beta",1e-3,1e-2, log=True)
    epsilon_max = trial.suggest_float("epsilon_max",0.5,0.9)

    seed = 43
    env = CropRotationEnv(seq_len=10, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 500
    dqn_agent = DeepQAgent(env = env,
                 number_hidden_units = number_hidden_units,
                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=lr, amsgrad=False, weight_decay = weight_decay),
                 batch_size = batch_size,
                 buffer_size = buffer_size,
                 alpha = alpha,
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, epsilon_max, 0.1, num_episodes),
                 delta_decay_schedule = lambda x: delta_decay_schedule(x, 1.0, 0.5, num_episodes),
                 gamma = 0.99,
                 update_frequency = 1,
                 seed= seed,
                 rule_options = "humus_and_breaks",
                 only_filter = False
                 )
    average_last_20_losses = 100.0
    average_last_20_rewards = 0.0
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state, filter_information = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        losses = []
        for t in count():
            action = dqn_agent.select_action(state, filter_information)
            observation, filter_information, reward, done, _ = env.step(action.item())
            
            # observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory
            avg_loss = dqn_agent.step(state, action, reward, next_state, done, filter_information, env)

            # Move to the next state
            losses.append(avg_loss)
            state = next_state
            total_reward += reward.item()
            if done:
                break
        average_last_20_losses = (average_last_20_losses * 19 + avg_loss) / 20
        average_last_20_rewards = (average_last_20_rewards * 19 + total_reward) / 20
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta}, Delta: {dqn_agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
    print('Complete')
    return math.log(average_last_20_losses), average_last_20_rewards

def run_optuna_study(num_episodes=500, n_trials=30, timeout=600):
    study = optuna.create_study(directions=["minimize","maximize"])
    # create partial function from objective function including num_episodes
    objective_partial = lambda trial: objective(trial, num_episodes)
    study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_avg_rewards = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest avg rewards: ")
    print(f"\tnumber: {trial_with_highest_avg_rewards.number}")
    print(f"\tvalues: {trial_with_highest_avg_rewards.values}")
    print(f"\tparams:")
    pp.pprint(trial_with_highest_avg_rewards.params)
    optuna.visualization.plot_pareto_front(study, target_names=["Average final losses", "Average final rewards"])

