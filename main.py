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
# env_testing_basic_prioritized.test_run(num_episodes = 1000)
# env_testing_basic_prioritized.run_optuna_study(num_episodes=500, n_trials=50, timeout=600)

# from experiments import env_testing_basic_sac
# env_testing_basic_sac.test_run(num_episodes = 2000)
# env_testing_basic_sac.run_optuna_study(num_episodes=500, n_trials=50, timeout=600)


######################################
# Advanced environment testing
######################################
# from experiments import env_testing_advanced_prioritized
# env_testing_advanced_prioritized.test_run(agent_type = "DQN_Prioritized")
# env_testing_advanced_prioritized.run_optuna_study(num_episodes=100, n_trials=50, timeout=600)

from experiments import env_testing_advanced_prioritized_symbolic
env_testing_advanced_prioritized_symbolic.test_run()
# env_testing_advanced_prioritized_symbolic.run_optuna_study()
# env_testing_advanced_prioritized_symbolic.check_filter()
