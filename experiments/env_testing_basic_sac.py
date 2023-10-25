from simulation_env.environment_basic.environment_basic import CropRotationEnv
from models.advanced.SAC import SACAgent
import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses_sac
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
TRAINING_EVALUATION_RATIO = 5


def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)

def test_run(num_episodes = 100):

    param_dict = {   'actor_lr': 7.160286161807239e-05,
    'batch_size': 142,
    'beta': 0.0012352161228288024,
    'buffer_size': 4726,
    'critic_lr': 0.0029202370240429274,
    'number_hidden_units': 135,
    'prio_alpha': 0.29137177035552214,
    'tau': 0.0012153367786148842,
    'temperature_initial': 0.6262300480955563,
    'weight_decay': 3.052537047433723e-05}

    seed = 43
    env = CropRotationEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac_agent = SACAgent(env = env,
                    number_hidden_units = param_dict["number_hidden_units"],
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["critic_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    # actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["actor_lr"]),
                    batch_size = param_dict["batch_size"],
                    buffer_size = param_dict["buffer_size"],
                    prio_alpha = param_dict["prio_alpha"],
                    temperature_initial = param_dict["temperature_initial"],
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, param_dict["beta"]),
                    tau = param_dict["tau"],
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, 0.7, 0.01, num_episodes),
                    seed= seed,
                    )
    rewards = []
    avg_critic1_losses = []
    avg_critic2_losses = []
    avg_actor_losses = []
    avg_temperature_losses = []
    for i_episode in range(num_episodes):
        evaluation_flag = i_episode % TRAINING_EVALUATION_RATIO == 0
        total_reward = 0
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        critic1_losses = []
        critic2_losses = []
        actor_losses = []
        temperature_losses = []
        for t in count():
            
            action = sac_agent.select_action(state, evaluation_flag=evaluation_flag)
            observation, reward, done, info = env.step(action)
            
            # observation, reward, done, truncated, _ = env.step(action.item())
            # Reward multiplication to compete with Entropy
            reward = torch.tensor([reward], device=device)*5.0
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory

            if not evaluation_flag:
                # critic_loss, critic2_loss, actor_loss, temperature_loss = sac_agent.step(state, action, reward, next_state, done)
                critic_loss, critic2_loss, actor_loss, temperature_loss = sac_agent.step(state, action, reward, next_state, done)
                critic1_losses.append(critic_loss)
                critic2_losses.append(critic2_loss)
                actor_losses.append(actor_loss)
                temperature_losses.append(temperature_loss)
            # Move to next state
            state = next_state
            numeric_reward = reward.item()*(max(env.cropYieldList.values())*1.2-env.negativeReward)/5.
            total_reward += numeric_reward
            crops_selected.append(info["Previous crop"])
            if done:
                break
        if not evaluation_flag:
            print(f"Run number: {i_episode}, Reward: {total_reward}, Temperature: {sac_agent._temperature}, Beta: {sac_agent.beta},  Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average critic2 loss: {torch.round(torch.tensor(critic2_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}, Average temperature loss: {torch.round(torch.tensor(temperature_losses).mean(),decimals=8)}")
            # print(f"Selected crops: {crops_selected}")
            avg_critic1_losses.append(torch.tensor(critic1_losses).mean())
            avg_critic2_losses.append(torch.tensor(critic2_losses).mean())
            avg_actor_losses.append(torch.tensor(actor_losses).mean())
            avg_temperature_losses.append(torch.tensor(temperature_losses).mean())
        else:
            print(f"Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            rewards.append(total_reward)
    print('Complete')
    plot_experiment(rewards)
    plot_losses_sac(avg_critic1_losses, avg_critic2_losses, avg_actor_losses)


def objective(trial, num_episodes = 1000):
    critic_lr = trial.suggest_float("critic_lr",1e-8,1e-2,log=True)
    actor_lr = trial.suggest_float("actor_lr",1e-8,1e-2,log=True)
    weight_decay = trial.suggest_float("weight_decay",1e-9,1e-1,log=True)
    batch_size = trial.suggest_int("batch_size", 32, 512, log=True)
    buffer_size = trial.suggest_int("buffer_size", 1000, 100000, log=True)
    number_hidden_units = trial.suggest_int("number_hidden_units", 32, 1024, log=True)
    rewards = []
    average_losses = []
    prio_alpha = trial.suggest_float("prio_alpha",0.1,0.9)
    temperature_initial = trial.suggest_float("temperature_initial",0.1,3.0)
    beta = trial.suggest_float("beta",1e-3,1e-2, log=True)
    tau = trial.suggest_float("tau",5e-4,0.3, log=True)

    seed = 43
    env = CropRotationEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac_agent = SACAgent(env = env,
                    number_hidden_units = number_hidden_units,
                    critic_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=critic_lr, amsgrad=False, weight_decay = weight_decay),
                    # actor_optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=param_dict["actor_lr"], amsgrad=False, weight_decay = param_dict["weight_decay"]),
                    # critic_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=param_dict["critic_lr"]),
                    actor_optimizer_fn = lambda parameters: optim.Adam(parameters, lr=actor_lr),
                    batch_size = batch_size,
                    buffer_size = buffer_size,
                    prio_alpha = prio_alpha,
                    temperature_initial = temperature_initial,
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                    tau = tau,
                    gamma = 0.99,
                    temperature_decay_schedule = lambda x: epsilon_decay_schedule(x, temperature_initial, 0.01, num_episodes),
                    seed= seed,
                    )
    rewards = []
    avg_critic1_losses = []
    avg_critic2_losses = []
    avg_actor_losses = []
    avg_temperature_losses = []
    average_last_20_critic1_losses = 100.0
    average_last_20_actor_losses = 0.0
    average_last_10_rewards = 0.0
    for i_episode in range(num_episodes):
        evaluation_flag = i_episode % TRAINING_EVALUATION_RATIO == 0
        total_reward = 0
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        critic1_losses = []
        actor_losses = []
        for t in count():

            action = sac_agent.select_action(state, evaluation_flag=evaluation_flag)
            observation, reward, done, info = env.step(action)
            
            # observation, reward, done, truncated, _ = env.step(action.item())
            # Reward multiplication to compete with Entropy
            reward = torch.tensor([reward], device=device)*5.0
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            if not evaluation_flag:
                # critic_loss, critic2_loss, actor_loss, temperature_loss = sac_agent.step(state, action, reward, next_state, done)
                critic_loss, critic2_loss, actor_loss, temperature_loss = sac_agent.step(state, action, reward, next_state, done)
                critic1_losses.append(critic_loss)
                actor_losses.append(actor_loss)
            numeric_reward = reward.item()*(max(env.cropYieldList.values())*1.2-env.negativeReward)/5.
            total_reward += numeric_reward
            # Move to next state
            state = next_state
            crops_selected.append(info["Previous crop"])
            if done:
                break
        if not evaluation_flag:
            avg_critic1_losses.append(torch.tensor(critic1_losses).mean())
            avg_actor_losses.append(torch.tensor(actor_losses).mean())
            average_last_20_critic1_losses = (average_last_20_critic1_losses * 19 + torch.tensor(critic1_losses).mean()) / 20
            average_last_20_actor_losses = (average_last_20_actor_losses * 19 + torch.tensor(actor_losses).mean()) / 20
            print(f"Learning: Reward: {total_reward}, Temperature: {sac_agent._temperature}, Average critic1 loss: {torch.round(torch.tensor(critic1_losses).mean(),decimals=8)}, Average actor loss: {torch.round(torch.tensor(actor_losses).mean(),decimals=8)}")
        else:
            print(f"Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            rewards.append(total_reward)    
            average_last_10_rewards = (average_last_10_rewards * 9 + total_reward) / 10


    return average_last_20_critic1_losses, average_last_20_actor_losses, average_last_10_rewards
                
    
def run_optuna_study(num_episodes=500, n_trials=30, timeout=600):
    study = optuna.create_study(directions=["minimize","minimize","maximize"])
    # create partial function from objective function including num_episodes
    objective_partial = lambda trial: objective(trial, num_episodes)
    study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    trial_with_highest_avg_rewards = max(study.best_trials, key=lambda t: t.values[2])
    print(f"Trial with highest avg rewards: ")
    print(f"\tnumber: {trial_with_highest_avg_rewards.number}")
    print(f"\tvalues: {trial_with_highest_avg_rewards.values}")
    print(f"\tparams:")
    pp.pprint(trial_with_highest_avg_rewards.params)
    optuna.visualization.plot_pareto_front(study, target_names=["Average final critic losses","Average final actor losses", "Average final rewards"])




