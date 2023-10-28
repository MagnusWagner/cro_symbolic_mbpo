from experiments.utilities import run_optuna_study, check_filter
# from experiments import env_testing_advanced
# env_testing_advanced.test_random()
# env_testing_advanced.test_print()
# env_testing_advanced.plot_prices_and_costs()

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
# env_testing_advanced_prioritized.test_run(num_episodes = 2000, seq_len = 5, seed = 43)
# run_optuna_study(agent_type = "prioritized", environment_type = "advanced", n_trials=30, timeout=1800, num_episodes = 1000, training_eval_ratio = 5, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 10, seed = 43)

# from experiments import env_testing_advanced_prioritized_symbolic
# # env_testing_advanced_prioritized_symbolic.test_run(num_episodes = 1000, DryWetInit = None, GroundTypeInit = None, deterministic = None, seq_len = 10, seed = 42, rule_options = "humus_and_breaks", only_filter = False) # "humus_and_breaks", # "only_break_rules_and_timing", "all"
# run_optuna_study(agent_type = "prioritized_symbolic", environment_type = "advanced", n_trials=30, timeout=1800, num_episodes = 1000, training_eval_ratio = 5, DryWetInit = None, GroundTypeInit = None, deterministic = None, rule_options = "humus_and_breaks", only_filter = False, seq_len = 5, seed = 43)
# check_filter(agent_type = "prioritized_symbolic")
