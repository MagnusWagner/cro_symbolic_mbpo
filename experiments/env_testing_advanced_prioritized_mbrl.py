import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import pprint
import optuna
from optuna.trial import TrialState
import math
from experiments.utilities import single_training_run, single_mbrl_training_run
# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def test_run(
        num_episodes = 500, 
        DryWetInit = None, 
        GroundTypeInit = None, 
        deterministic = None, 
        seq_len = 10,
        seed = 43
        ):
    param_dict = {   'alpha': 0.55,
    'batch_size': 512,
    'beta': 0.01,
    'buffer_size': 2000,
    'epsilon_max': 0.8,
    'lr': 2.44e-08,
    'number_hidden_units': 1024,
    'tau': 0.3,
    'weight_decay': 0.00002,
    'num_dynamics_model_training_steps': 25,
    'dynamics_model_batch_size': 256,
    'num_rollouts':10,
    'rollout_length':1,
    'agent_training_steps':1}
    
    
    training_eval_ratio = 10
    single_training_run(
        param_dict,
        agent_type= "prioritized", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        num_episodes = num_episodes,
        training_eval_ratio = training_eval_ratio,
        # Environment parameters
        DryWetInit = DryWetInit,
        GroundTypeInit = GroundTypeInit,
        deterministic = deterministic,
        seq_len = seq_len,
        mbrl_flag = True,
        seed = seed)


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
    epsilon_max = trial.suggest_float("epsilon_max",0.2,0.9)
    tau = trial.suggest_float("tau",0.01,0.5)

    seed = 43
    env = CropRotationEnv(seq_len=10, seed = seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQN_Prioritized(env = env,
                    number_hidden_units = number_hidden_units,
                    optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=lr, amsgrad=False, weight_decay = weight_decay),
                    batch_size = batch_size,
                    buffer_size = buffer_size,
                    alpha = alpha,
                    beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, beta),
                    epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, epsilon_max, 0.01, num_episodes),
                    gamma = 0.99,
                    tau = tau,
                    seed= seed,
                    )
    average_last_20_losses = 100.0
    average_last_10_rewards = 0.0
    for i_episode in range(num_episodes):
        evaluation_flag = i_episode % TRAINING_EVALUATION_RATIO == 0
        total_reward = 0
        # Initialize the environment and get it's state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        crops_selected = []
        losses = []
        for t in count():
            action = dqn_agent.select_action(state, evaluation_flag = evaluation_flag)
            observation, _, reward, done, info = env.step(action)
            
            # observation, reward, done, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory
            if not evaluation_flag:
                avg_loss = dqn_agent.step(state, action, reward, next_state, done)
            # Move to the next state
            state = next_state
            total_reward += reward.item()*(env.max_reward-env.min_reward)
            crops_selected.append(info["Previous crop"])
            if done:
                break
        if not evaluation_flag:
            print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta},  Average loss: {torch.tensor(losses).mean()}")
            average_last_20_losses = (average_last_20_losses * 19 + avg_loss) / 20
        else:
            print(f"#{i_episode}, Evaluation: Reward: {total_reward}. Selected crops: {crops_selected}")
            average_last_10_rewards = (average_last_10_rewards * 9 + total_reward) / 10
    


    return math.log(average_last_20_losses), average_last_10_rewards

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