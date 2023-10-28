import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import pprint
import optuna
from optuna.trial import TrialState
from numpy import random
import math
from experiments.utilities import single_training_run

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)
TRAINING_EVALUATION_RATIO = 5


def test_run(num_episodes = 500, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 5, seed = 43, rule_options = "humus_and_breaks", only_filter = False):
    param_dict = {   
    'alpha': 0.7694843699721963,
    'batch_size': 350,
    'beta': 0.005011142099839954,
    'buffer_size': 5111,
    'delta_max': 0.98,
    'epsilon_max': 0.3661880105349842,
    'lr': 0.003261362212533657,
    'number_hidden_units': 558,
    'tau': 0.07316558231562796,
    'weight_decay': 0.014870085686472516}
    training_eval_ratio = 5
    single_training_run(
        param_dict,
        agent_type = "prioritized_symbolic", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "advanced",
        rule_options = rule_options,
        only_filter = only_filter,
        num_episodes = num_episodes,
        training_eval_ratio = training_eval_ratio,
        DryWetInit = DryWetInit,
        GroundTypeInit = GroundTypeInit,
        deterministic = deterministic,
        seq_len = seq_len,
        seed = seed)


