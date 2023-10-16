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

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def test_print():
    # Initialize crop rotation environment
    env = CropRotationEnv()
    env.render()
    
def test_random():
    env = CropRotationEnv()
    # Generate 5 random crop rotations without training (for enviornment testing)
    episodes = 100
    for episode in range(1, episodes+1):
        state, filter_information = env.reset()
        done = False
        score = 0 
        while not done:
            action = env.action_space.sample()
            observation, filter_information, reward, done, _ = env.step(action)
            score+=reward
            # pp.pprint(info)

        print('Episode:{} Score:{}'.format(episode, score))


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
    count_broken_rules = 0
    no_possible_actions = 0
    for episode in range(1, num_episodes+1):
        state, filter_information = env.reset()
        done = False
        score = 0 
        while not done:
            possible_actions = dqn_agent.filter_actions(filter_information)
            if possible_actions:
                action = np.random.choice(possible_actions)
                observation, filter_information, reward, done, info = env.step(action)
                n_broken_rules.append(info["Num broken rules"])
                if info["Num broken rules"] > 0:
                    count_broken_rules += 1
            else:
                action = env.action_space.sample()
                print(f"No possible action. Taking random action: {action}.")
                no_possible_actions += 1
                observation, filter_information, reward, done, info = env.step(action)
    print("Number of events without possible actions:" , no_possible_actions)
    print("% of events with possible actions but broken rules:", count_broken_rules / (len(n_broken_rules) + no_possible_actions) * 100)
    # Plot the number of broken rules for each episode
    import matplotlib.pyplot as plt
    plt.plot(n_broken_rules)
    plt.title('Number of broken rules')
    plt.xlabel('Episode')
    plt.ylabel('Number of broken rules')
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
    
    rewards = []
    average_losses = []

    seed = 43
    env = CropRotationEnv(seq_len=10, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    dqn_agent = DeepQAgent(env = env,
                 number_hidden_units = 1024,
                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=2e-6, amsgrad=False, weight_decay = 1e-2),
                 batch_size = 512,
                 buffer_size = 100000,
                 alpha = 0.4,
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, 2e-3),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, 0.5, 0.1, num_episodes),
                 delta_decay_schedule = lambda x: delta_decay_schedule(x, 1.0, 0.5, num_episodes),
                 gamma = 0.99,
                 update_frequency = 1,
                 seed= seed,
                 rule_options = "only_break_rules_and_timing",
                 only_filter = False
                 )

    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state, filter_information = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            losses = []
            action = dqn_agent.select_action(state, filter_information)
            observation, filter_information, reward, done, _ = env.step(action.item())
            
            # observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory
            avg_loss = dqn_agent.step(state, action, reward, next_state, done)

            # Move to the next state
            losses.append(avg_loss)
            state = next_state
            total_reward += reward.item()
            if done:
                break
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta}, Delta: {dqn_agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
    total_reward = 0
    # Initialize the environment and get it's state
    state, filter_information = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    crops_selected = []
    total_reduction_factors = []
    for t in count():
        losses = []
        action = dqn_agent.select_action(state, filter_information, greedy=True)
        observation, filter_information, reward, done, info = env.step(action.item())
        total_reward += reward
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
    plot_losses(average_losses)




    # reward_list_average = run_experiment(env, dqn_agent, steps)
    # plot_experiment(steps, env.cropRotationSequenceLengthStatic, reward_list_average)

