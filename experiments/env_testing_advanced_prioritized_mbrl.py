import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import pprint
import optuna
from optuna.trial import TrialState
import math
from experiments.utilities import single_training_run
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
        seed = 43,
        ):
    param_dict = {   
    'batch_size': 256,
    'beta': 0.00575,
    'buffer_size': 10000,
    'num_dynamics_model_training_steps': 10,
    'dynamics_model_batch_size': 256,
    'rollout_length': 1,
    'delta_max': 0.9,
    'agent_training_steps': 1,
    'alpha': 0.8012716196920497,
    'epsilon_max': 0.827649425370713,
    'lr': 9.89153562086848e-07,
    'num_rollouts': 17,
    'number_hidden_units': 360,
    'tau': 0.060914141743975185,
    'weight_decay': 1.7207443066544826e-08}
    
    
    
    training_eval_ratio = 5
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
        pretrain_flag = True,
        pretrain_buffer_size = 50000,
        pretrain_num_steps = 1000,
        seed = seed,
        plot_flag = True)