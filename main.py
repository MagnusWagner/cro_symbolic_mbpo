from experiments.hyperparam_optimization import run_optuna_study
from experiments.utilities import check_filter
# from experiments import env_testing_advanced
# env_testing_advanced.test_random()
# env_testing_advanced.test_print()
# env_testing_advanced.plot_prices_and_costs()
# env_testing_advanced.test_model_prediction_with_random_actions(seed = 42, batch_size = 128, num_train_episodes = 2000, num_model_training_steps = 5)

######################################
# Gym testing
######################################
# from experiments import gym_testing_sac
# gym_testing_sac.test_run(num_episodes = 500)


######################################
# Basic environment testing
######################################
# from experiments import env_testing_basic_prioritized
# env_testing_basic_prioritized.test_run(num_episodes = 2000, seq_len = 5, seed = 43)
# run_optuna_study(agent_type = "prioritized", environment_type = "basic", n_trials=3, timeout=1200, num_episodes = 100, training_eval_ratio = 5, seq_len = 5, seed = 43)

# from experiments import env_testing_basic_sac
# env_testing_basic_sac.test_run(num_episodes = 500, seq_len = 5, seed = 43)
# run_optuna_study(agent_type = "sac", environment_type = "basic", n_trials=20, timeout=1200, num_episodes = 1000, training_eval_ratio = 5, seq_len = 5, seed = 43)

######################################
# Advanced environment testing
######################################
# from experiments import env_testing_advanced_prioritized
# env_testing_advanced_prioritized.test_run(num_episodes = 300, DryWetInit = 1.0, GroundTypeInit = 1.0, deterministic = True, seq_len = 10, seed = 42)
# run_optuna_study(agent_type = "prioritized", environment_type = "advanced", n_trials=30, timeout=1800, num_episodes = 1000, training_eval_ratio = 5, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 10, seed = 43)

# from experiments import env_testing_advanced_sac
# env_testing_advanced_sac.test_run(num_episodes = 500, DryWetInit = 1.0, GroundTypeInit = 1.0, deterministic = True, seq_len = 10, seed = 42)
# run_optuna_study(agent_type = "sac", environment_type = "advanced", n_trials=30, timeout=1800, num_episodes = 1000, training_eval_ratio = 5, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 10, seed = 43)

######################################
# Advanced environment symbolic testing
######################################
# from experiments import env_testing_advanced_prioritized_symbolic
# env_testing_advanced_prioritized_symbolic.test_run(
#     num_episodes = 300, 
#     DryWetInit = 1.0, 
#     GroundTypeInit = 1.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     rule_options = "humus_and_breaks", 
#     only_filter = False) # "humus_and_breaks", # "only_break_rules_and_timing", "all"
# run_optuna_study(agent_type = "prioritized_symbolic", environment_type = "advanced", n_trials=30, timeout=1800, num_episodes = 1000, training_eval_ratio = 5, DryWetInit = None, GroundTypeInit = None, deterministic = None, rule_options = "humus_and_breaks", only_filter = False, seq_len = 5, seed = 43)
# check_filter(agent_type = "prioritized_symbolic")

# from experiments import env_testing_advanced_sac_symbolic
# env_testing_advanced_sac_symbolic.test_run(
#     num_episodes = 200, 
#     DryWetInit = 1.0, 
#     GroundTypeInit = 1.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     rule_options = "humus_and_breaks", 
#     only_filter = False)

# Only Filter
# from experiments import env_testing_advanced_prioritized_symbolic
# env_testing_advanced_prioritized_symbolic.test_run(
#     num_episodes = 200, 
#     DryWetInit = 0.0, 
#     GroundTypeInit = 0.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     rule_options = "humus_and_breaks", 
#     only_filter = True)

# Random Run
# from experiments import env_testing_advanced_prioritized
# env_testing_advanced_prioritized.test_run(
#     num_episodes = 200, 
#     DryWetInit = 0.0, 
#     GroundTypeInit = 0.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     random_flag = True)

### MBRL
###############################
# DO NOT RUN AGAIN
# from experiments import env_testing_advanced_prioritized_mbrl
# env_testing_advanced_prioritized_mbrl.test_run(
#         num_episodes = 200, 
#         DryWetInit = 1.0, 
#         GroundTypeInit = 1.0, 
#         deterministic = True, 
#         seq_len = 10,
#         seed = 42
#         )

# DO NOT RUN AGAIN
# from experiments import env_testing_advanced_sac_mbrl
# env_testing_advanced_sac_mbrl.test_run(
#         num_episodes = 200, 
#         DryWetInit = 1.0, 
#         GroundTypeInit = 1.0, 
#         deterministic = True, 
#         seq_len = 10,
#         seed = 42
#         )

# DO NOT RUN AGAIN
# from experiments import env_testing_advanced_prioritized_symbolic_mbrl
# env_testing_advanced_prioritized_symbolic_mbrl.test_run(
#         num_episodes = 100, 
#         DryWetInit = 1.0, 
#         GroundTypeInit = 1.0, 
#         deterministic = True, 
#         seq_len = 10,
#         seed = 42,
#         rule_options = "humus_and_breaks",
#         only_filter = False
#         )


# from experiments import env_testing_advanced_sac_symbolic_mbrl
# env_testing_advanced_sac_symbolic_mbrl.test_run(
#         num_episodes = 200, 
#         DryWetInit = 1.0, 
#         GroundTypeInit = 1.0, 
#         deterministic = True, 
#         seq_len = 10,
#         seed = 42,
#         rule_options = "humus_and_breaks",
#         only_filter = False
#         )

# param_dict = {   
#     'batch_size': 256,
#     'beta': 0.00575,
#     'buffer_size': 10000,
#     'num_dynamics_model_training_steps': 10,
#     'dynamics_model_batch_size': 256,
#     'rollout_length': 1,
#     'agent_training_steps': 1,
#    'actor_lr': 9.638988396090234e-07,
#     'critic_lr': 1.306628936455211e-08,
#     'num_rollouts': 16,
#     'number_hidden_units': 225,
#     'prio_alpha': 0.2814932040505599,
#     'tau': 0.15056571804326951,
#     'temperature_initial': 1.5661943042671909,
#     'weight_decay': 1.299881320154841e-09}

# run_optuna_study(
#         agent_type = "sac", 
#         environment_type = "advanced", 
#         mbrl_flag = True, 
#         n_trials=2000, 
#         timeout=7200, 
#         num_episodes = 100, 
#         training_eval_ratio = 10, 
#         DryWetInit = 1.0, 
#         GroundTypeInit = 1.0, 
#         deterministic = True, 
#         rule_options = "humus_and_breaks", 
#         only_filter = False, 
#         seq_len = 10,
#         pretrain_buffer_size = 5000,
#         pretrain_num_steps = 500,
#         param_dict = param_dict,
#         seed = 43)


### Neighbourhood experiments
# from experiments import env_testing_advanced_prioritized_neighbour
# env_testing_advanced_prioritized_neighbour.test_run(num_episodes = 300, DryWetInit = 1.0, GroundTypeInit = 1.0, deterministic = True, seq_len = 10, seed = 42, rule_options = "humus_and_breaks")

# param_dict = {   
#     'batch_size': 256,
#     'buffer_size': 10000}

# run_optuna_study(
#     agent_type = "sac_symbolic", 
#     environment_type = "advanced", 
#     n_trials=1000, 
#     timeout=14400, 
#     num_episodes = 200, 
#     training_eval_ratio = 5, 
#     DryWetInit = 1.0, 
#     GroundTypeInit = 1.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 43,
#     neighbour_flag = True,
#     num_neighbours = 50,
#     neighbour_buffer_size = 5000,
#     rule_options = "humus_and_breaks",
#     param_dict = param_dict,
#     )


# from experiments import env_testing_advanced_prioritized_symbolic_neighbour
# env_testing_advanced_prioritized_symbolic_neighbour.test_run(
#     num_episodes = 300, 
#     DryWetInit = 1.0, 
#     GroundTypeInit = 1.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     rule_options = "humus_and_breaks", 
#     only_filter = False) # "humus_and_breaks", # "only_break_rules_and_timing", "all"

# from experiments import env_testing_advanced_sac_neighbour
# env_testing_advanced_sac_neighbour.test_run(num_episodes = 200, DryWetInit = 1.0, GroundTypeInit = 1.0, deterministic = True, seq_len = 10, seed = 42, rule_options = "humus_and_breaks")

# from experiments import env_testing_advanced_sac_symbolic_neighbour
# env_testing_advanced_sac_symbolic_neighbour.test_run(
#     num_episodes = 200, 
#     DryWetInit = 1.0, 
#     GroundTypeInit = 1.0, 
#     deterministic = True, 
#     seq_len = 10, 
#     seed = 42, 
#     rule_options = "humus_and_breaks", 
#     only_filter = False)


from experiments.evaluations.gather_evaluation_data import gather_evaluation_data, gather_pretrained_evaluation_data
gather_evaluation_data()
# gather_pretrained_evaluation_data()