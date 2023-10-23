from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
from models.advanced.SAC import SACAgent
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

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

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
    env = CropRotationEnv(seq_len=5, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 10000
    dqn_agent = SACAgent(env = env,
                 _number_hidden_units = param_dict["number_hidden_units"],
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
                 rule_options = "only_break_rules_and_timing",
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
        for t in count():
            losses = []
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
        print(f"Filtered difference: {filter_dict[True].mean() - filter_dict[False].mean()}")
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
        action,filtered_flag = dqn_agent.select_action(state, filter_information, greedy=True)
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

def objective(trial):
    # max_opt_steps = 10
    lr = trial.suggest_float("lr",1e-8,1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-8,1e-1,log=True)
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
                 rule_options = "only_break_rules_and_timing",
                 only_filter = False
                 )
    # average_last_ten_rewards = 0.0
    average_last_ten_losses = 100.0
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
        # average_last_ten_rewards = (average_last_ten_rewards * 9 + total_reward) / 10
        average_last_ten_losses = (average_last_ten_losses * 9 + avg_loss) / 10
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta}, Delta: {dqn_agent.delta_threshold},  Avg loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
        # trial.report(average_last_ten_rewards, i_episode)
        trial.report(average_last_ten_losses, i_episode)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    print('Complete')
    # return average_last_ten_rewards
    return average_last_ten_losses

def run_optuna_study():
    # study = optuna.create_study(direction="maximize")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

