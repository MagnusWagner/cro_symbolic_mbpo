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
        only_filter = False        
        ):
    param_dict = {   
    'batch_size': 256,
    'beta': 0.00575,
    'buffer_size': 10000,
    'num_dynamics_model_training_steps': 10,
    'dynamics_model_batch_size': 256,
    'rollout_length': 1,
    'actor_lr': 8.406501021342772e-05,
    'agent_training_steps': 1,
    'critic_lr': 9.535951875666365e-05,
    'delta_max': 0.8667731233081759,
    'num_rollouts': 15,
    'number_hidden_units': 246,
    'prio_alpha': 0.3593041553249767,
    'tau': 0.13355766080300563,
    'temperature_initial': 0.5791068472658799,
    'weight_decay': 0.006974057338183232}
    
    
    training_eval_ratio = 5
    single_training_run(
        param_dict,
        agent_type = "sac_symbolic", #["prioritized","sac","prioritized_symbolic","sac_symbolic"]
        environment_type = "advanced",
        rule_options = rule_options,
        only_filter = only_filter,
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
        plot_flag = True,
        seed = seed)