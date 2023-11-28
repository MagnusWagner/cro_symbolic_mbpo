import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import pprint
from experiments.utilities import single_training_run

# Create pprinter
pp = pprint.PrettyPrinter(indent=4)


def test_run(num_episodes = 500, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 5, seed = 43, rule_options = "humus_and_breaks", only_filter = False):
    param_dict = {   
    'batch_size': 256,
    'beta': 0.00575,
    'buffer_size': 10000,
    'alpha': 0.2891978839719468,
    'neighbour_alpha': 1.0,
    'delta_max': 0.8905885911276994,
    'epsilon_max': 0.8037188702073559,
    'lr': 8.351864571435923e-05,
    'number_hidden_units': 417,
    'tau': 0.35702502666141533,
    'weight_decay': 1.040831556255643e-05}
    
    
    
    # {   
    # 'alpha': 0.5554887943329508,
    # 'batch_size': 381,
    # 'beta': 0.0010198361240556146,
    # 'buffer_size': 5212,
    # 'epsilon_max': 0.7954054344061976,
    # 'delta_max': 0.98,
    # 'lr': 2.4401561648489245e-05,
    # 'number_hidden_units': 926,
    # 'tau': 0.2906968953074242,
    # 'weight_decay': 0.0019988460143838633}
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
        seed = seed,
        neighbour_flag = True,
        num_neighbours = 50,
        neighbour_buffer_size = 5000,
        plot_flag = True)


