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
        rule_options = "humus_and_breaks", 
        only_filter = False):    
    param_dict = {   
        'batch_size': 256,
        'beta': 0.00575,
        'buffer_size': 10000,
        'actor_lr': 0.0029173695676940102,
        'critic_lr': 4.648702385874135e-05,
        'delta_max': 0.370607900987715,
        'number_hidden_units': 652,
        'prio_alpha': 0.8987304835041263,
        'tau': 0.257536769088055,
        'temperature_initial': 0.7017463941895495,
        'weight_decay': 0.02684551375251473}
        
    
    
    # {   
    #     'actor_lr': 0.00013,
    #     'batch_size': 413,
    #     'beta': 0.00575,
    #     'buffer_size': 7303,
    #     'critic_lr': 0.00121,
    #     'number_hidden_units': 1024, #'number_hidden_units': 700,
    #     'delta_max': 0.98,
    #     'prio_alpha': 0.774,
    #     'tau': 0.0234,
    #     'temperature_initial': 0.402,
    #     'weight_decay': 0.00163
    #     }

    training_eval_ratio = 5
    single_training_run(param_dict,
        agent_type = "sac_symbolic", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "advanced",
        rule_options = rule_options,
        only_filter = only_filter,
        num_episodes = num_episodes,
        training_eval_ratio = training_eval_ratio,
        DryWetInit = DryWetInit,
        GroundTypeInit = GroundTypeInit,
        deterministic = deterministic,
        seq_len = seq_len,
        seed = seed,
        plot_flag = True)