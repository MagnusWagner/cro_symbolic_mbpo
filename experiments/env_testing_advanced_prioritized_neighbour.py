import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import pprint
from experiments.utilities import single_training_run

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def test_run(num_episodes = 500, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 5, seed = 43, rule_options = "humus_and_breaks"):
    param_dict = {   
    'batch_size': 256,
    'buffer_size': 10000,
    'epsilon_max': 0.72,
    'number_hidden_units': 784,
    'alpha': 0.28311782705908295,
    'beta': 0.0014143082163620756,
    'lr': 1.90140826666231e-05,
    'neighbour_alpha': 0.47595536538391847,
    'tau': 0.2631064536553188,
    'weight_decay': 6.03071565279281e-07}
    
    
    
    # old config
    # {   'alpha': 0.55,
    # 'batch_size': 381,
    # 'beta': 0.00101,
    # 'buffer_size': 5000,
    # 'epsilon_max': 0.8,
    # 'lr': 2.44e-05,
    # 'number_hidden_units': 926,
    # 'tau': 0.3,
    # 'weight_decay': 0.002}
    training_eval_ratio = 5
    single_training_run(
        param_dict,
        agent_type = "prioritized", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "advanced",
        rule_options = rule_options,
        num_episodes = num_episodes,
        training_eval_ratio = training_eval_ratio,
        DryWetInit = DryWetInit,
        GroundTypeInit = GroundTypeInit,
        deterministic = deterministic,
        seq_len = seq_len,
        seed = seed,
        neighbour_flag = True,
        num_neighbours = 50,
        neighbour_buffer_size = 10000,
        plot_flag = True)
