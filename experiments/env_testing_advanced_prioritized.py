from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
from models.basic.DQN_Prioritized import DeepQAgent as DQN_Prioritized
# from models.basic.DQNPytorch import DeepQAgent as DQN
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
import math

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def test_print():
    # Initialize crop rotation environment
    env = CropRotationEnv()
    env.render()
    
def test_random():
    env = CropRotationEnv()
    # Generate 5 random crop rotations without training (for environment testing)
    episodes = 100
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        while not done:
            action = env.action_space.sample()
            n_state, _, reward, done, info = env.step(action)
            score+=reward
            # pp.pprint(info)

        print('Episode:{} Score:{}'.format(episode, score))



def constant_annealing_schedule(n, constant):
    return constant


def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def test_run(agent_type = "DQN" # "DQN", "DQN_Prioritized"
             ):
    # max_opt_steps = 10
    
    
    rewards = []
    average_losses = []

    param_dict = {   'alpha': 0.6682581518763132,
    'batch_size': 512,
    'beta': 0.007583023190822477,
    'buffer_size': 15756,
    'epsilon_max': 0.8765399831991354,
    'lr': 3.595681414558709e-08,
    'number_hidden_units': 32,
    'weight_decay': 5.796144878637577e-05}
    rewards = []
    average_losses = []

    seed = 43
    env = CropRotationEnv(seq_len=10, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    if agent_type == "DQN_Prioritized":
        dqn_agent = DQN_Prioritized(env = env,
                    number_hidden_units = param_dict["number_hidden_units"],
                    optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    batch_size = param_dict["batch_size"],
                    buffer_size = param_dict["buffer_size"],
                    alpha = param_dict["alpha"],
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                    epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, param_dict["epsilon_max"], 0.1, num_episodes),
                    gamma = 0.99,
                    update_frequency = 1,
                    seed= seed,
                    )
    else:
        print("No agent selected.")
        raise NotImplementedError

    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        for t in count():
            losses = []
            action = dqn_agent.select_action(state)
            observation, _, reward, done, info = env.step(action.item())
            
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
            crops_selected.append(info["Previous crop"])
            if done:
                break
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta},  Average loss: {torch.tensor(losses).mean()}")
        print(f"Selected crops: {crops_selected}")
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
        action = dqn_agent.select_greedy_action(state)
        observation, _, reward, done, info = env.step(action.item())
        total_reward += reward
        crops_selected.append(info["Previous crop"])
        total_reduction_factors.append(info["Total Reduction Factor"])
        # observation, reward, done, truncated, _ = env.step(action.item())
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


def objective(trial):
    # max_opt_steps = 10
    lr = trial.suggest_float("lr",1e-8,1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-9,1e-1,log=True)
    batch_size = trial.suggest_int("batch_size", 32, 1024, log=True)
    buffer_size = trial.suggest_int("buffer_size", 1000, 100000, log=True)
    number_hidden_units = trial.suggest_int("number_hidden_units", 32, 1024, log=True)
    rewards = []
    average_losses = []
    alpha = trial.suggest_float("alpha",0.1,0.9)
    beta = trial.suggest_float("beta",1e-3,1e-2, log=True)
    epsilon_max = trial.suggest_float("epsilon_max",0.2,0.9)

    seed = 43
    env = CropRotationEnv(seq_len=5, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    dqn_agent = DQN_Prioritized(env = env,
                 number_hidden_units = number_hidden_units,
                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=lr, amsgrad=False, weight_decay = weight_decay),
                 batch_size = batch_size,
                 buffer_size = buffer_size,
                 alpha = alpha,
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, epsilon_max, 0.1, num_episodes),
                 gamma = 0.99,
                 seed= seed,
                 )
    average_last_ten_losses = 100.0
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            losses = []
            action = dqn_agent.select_action(state)
            observation, _, reward, done, _ = env.step(action.item())
            
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
        average_last_ten_losses = (average_last_ten_losses * 9 + avg_loss) / 10
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta},  Avg loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
        trial.report(math.log(average_last_ten_losses), i_episode)
        if trial.should_prune() and i_episode > 200:
            raise optuna.exceptions.TrialPruned()
    print('Complete')
    return math.log(average_last_ten_losses)

def run_optuna_study():
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
    pp.pprint(trial.params)

