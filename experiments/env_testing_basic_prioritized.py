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


def test_run(num_episodes = 500, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 5, seed = 43):
    param_dict = {   'alpha': 0.4,
    'batch_size': 128,
    'beta': 0.00145,
    'buffer_size': 10000,
    'epsilon_max': 0.5,
    'lr': 1e-4,
    'number_hidden_units': 226,
    'weight_decay': 0.0,
    'tau' : 0.05}
    training_eval_ratio = 5
    single_training_run(
        param_dict,
        agent_type = "prioritized", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "basic",
        rule_options = None,
        num_episodes = num_episodes,
        training_eval_ratio = training_eval_ratio,
        DryWetInit = None,
        GroundTypeInit = None,
        deterministic = None,
        seq_len = seq_len,
        seed = seed)
