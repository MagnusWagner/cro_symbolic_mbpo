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
    'agent_training_steps': 5,
    'alpha': 0.439459983449515,
    'delta_max': 0.3128660150436883,
    'epsilon_max': 0.5177025239832722,
    'lr': 0.0017246297421046532,
    'num_rollouts': 18,
    'number_hidden_units': 569,
    'tau': 0.011957482344885119,
    'weight_decay': 3.856900160405226e-06}
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
        mbrl_flag = True,
        pretrain_flag = True,
        pretrain_buffer_size = 50000,
        pretrain_num_steps = 1000,
        seed = seed)

