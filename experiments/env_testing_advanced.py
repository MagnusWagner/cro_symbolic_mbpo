from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
import numpy as np



def test_print():
    # Initialize crop rotation environment
    env = CropRotationEnv()
    env.render()
    
def test_random():
    env = CropRotationEnv()
    # Generate 5 random crop rotations without training (for enviornment testing)
    episodes = 100
    for episode in range(1, episodes+1):
        state, filter_information = env.reset()
        done = False
        score = 0 
        while not done:
            action = env.action_space.sample()
            observation, filter_information, reward, done, _ = env.step(action)
            score+=reward
            # pp.pprint(info)

        print('Episode:{} Score:{}'.format(episode, score))


def plot_prices_and_costs():
    env = CropRotationEnv(seed = 43, seq_len = 10)
    # Generate 5 random crop rotations without training (for enviornment testing)

    state, filter_information = env.reset()
    prices = env.prices.reshape((1,len(env.prices)))
    sowing_costs = env.sowing_costs.reshape((1,len(env.sowing_costs)))
    other_costs = env.other_costs.reshape((1,len(env.other_costs)))
    N_costs = np.array([env.N_costs]).reshape((1,1))
    P_costs = np.array([env.P_costs]).reshape((1,1))
    K_costs = np.array([env.K_costs]).reshape((1,1))
    done = False
    while not done:
        action = env.action_space.sample()
        observation, filter_information, reward, done, _ = env.step(action)
        prices = np.vstack((prices, env.prices.reshape((1,len(env.prices)))))
        sowing_costs = np.vstack((sowing_costs, env.sowing_costs.reshape((1,len(env.sowing_costs)))))
        other_costs = np.vstack((other_costs, env.other_costs.reshape((1,len(env.other_costs)))))
        # Calculate N_costs, P_costs and K_costs which are not arrays but floats
        N_costs_tmp = np.array([env.N_costs])
        P_costs_tmp = np.array([env.P_costs])
        K_costs_tmp = np.array([env.K_costs])
        N_costs = np.vstack((N_costs, N_costs_tmp.reshape((1,1))))
        P_costs = np.vstack((P_costs, P_costs_tmp.reshape((1,1))))
        K_costs = np.vstack((K_costs, K_costs_tmp.reshape((1,1))))
    first_prices = prices[0]
    first_sowing_costs = sowing_costs[0]
    first_other_costs = other_costs[0]
    first_N_costs = N_costs[0]
    first_P_costs = P_costs[0]
    first_K_costs = K_costs[0]
    # Calculate normalized prices and costs
    normalized_prices = (prices-first_prices) / first_prices
    normalized_sowing_costs = (sowing_costs-first_sowing_costs) / first_sowing_costs
    normalized_other_costs = (other_costs-first_other_costs) / first_other_costs
    normalized_N_costs =  (N_costs-first_N_costs) / first_N_costs
    normalized_P_costs = (P_costs-first_P_costs) / first_P_costs
    normalized_K_costs = (K_costs-first_K_costs) / first_K_costs
    # Plot each price and cost variable over time where each row (first dimension) represents a time point and each column (second dimension) represents a different crop (indices 0 to 22)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(6, 1, figsize=(10, 10))
    axs[0].plot(normalized_prices)
    axs[0].set_title('Normalized prices')
    axs[1].plot(normalized_sowing_costs)
    axs[1].set_title('Normalized sowing costs')
    axs[2].plot(normalized_other_costs)
    axs[2].set_title('Normalized other costs')
    axs[3].plot(normalized_N_costs)
    axs[3].set_title('Normalized N costs')
    axs[4].plot(normalized_P_costs)
    axs[4].set_title('Normalized P costs')
    axs[5].plot(normalized_K_costs)
    axs[5].set_title('Normalized K costs')
    for ax in axs.flat:
        ax.set(xlabel='Crop', ylabel='Normalized value')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()