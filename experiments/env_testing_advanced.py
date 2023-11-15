from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
import numpy as np
import math
from models.utilities.ReplayBufferPrioritized import UniformReplayBuffer, Experience
import torch 
from models.advanced.fake_env import FakeEnv
from models.advanced.model_utilities import format_samples_for_training, create_full_replay_buffer
from tqdm import tqdm


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


def test_model_prediction_with_random_actions(seed = 42, batch_size = 128, num_train_episodes = 1000, num_model_training_steps = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CropRotationEnv(seed = seed, seq_len = 10)
    reward_factor = 5.0/(env.max_reward-env.min_reward)
    fake_env = FakeEnv(device = device, seed = seed)
    num_models = fake_env.get_num_models()
    model_keys = fake_env.get_model_keys()
    
    random_state = np.random.RandomState(seed)
    experience_replay_buffer = UniformReplayBuffer(batch_size = 128,
                                        buffer_size = 10000,
                                        random_state=random_state)
    
    # Create test set
    test_set_size = 500
    test_replay_buffer = create_full_replay_buffer(length = test_set_size, seq_len=10, seed = seed+1, batch_size = 128, device = device)
    _, test_experiences = test_replay_buffer.uniform_sample(replace = False,batch_size=test_set_size)
    states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*test_experiences))
    test_inputs, test_outputs = format_samples_for_training(
        states=states, 
        actions = actions, 
        rewards = rewards, 
        next_states = next_states,
        device = device,
        num_actions = env.num_crops,
        )

    all_mean_mse_losses = {}
    all_mean_kl_losses = {}
    for model_key in model_keys:
        all_mean_mse_losses[model_key] = []
        all_mean_kl_losses[model_key] = []
    evaluation_flag = False
    for episode in tqdm(range(1, num_train_episodes+1)):
        evaluation_flag = True if episode % 10 == 0 else False
        # Gather actual experience
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = env.action_space.sample()
            observation, _, reward, done, _ = env.step(action)
            # pp.pprint(info)
            if done:
                break
            if observation is None:
                print("next_state is None")

            reward_tensor = torch.tensor([reward], device=device)*reward_factor
            next_state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            action_tensor = torch.tensor([action]).to(device)
            experience = Experience(state_tensor, action_tensor.view(1,1), reward_tensor.view(1,1), next_state_tensor, torch.tensor([done]).view(1,1))
            # print("Stats:", replay_buffer.buffer_size, replay_buffer.batch_size, replay_buffer._buffer_length, replay_buffer._current_idx)
            experience_replay_buffer.add(experience)
            state = next_state_tensor
        if episode > 0:
            # Train dynamics model
            fake_env.train(
                replay_buffer = experience_replay_buffer, 
                num_steps = num_model_training_steps, 
                batch_size = min(experience_replay_buffer.buffer_length, batch_size)
            )
            if evaluation_flag:
                mean_mse_losses, mean_kl_losses = fake_env.eval(test_inputs, test_outputs)
                for model_key in model_keys:
                    all_mean_mse_losses[model_key].append(np.log(mean_mse_losses[model_key]))
                    all_mean_kl_losses[model_key].append(np.log(mean_kl_losses[model_key]))
            # Test dynamics model
            model_idxs = []
            for idx, model_key in enumerate(model_keys):
                model_idx = random_state.randint(0, num_models[idx])
                model_idxs.append(model_idx)

            dynamics_next_state, dynamics_reward = fake_env.predict(
                state = state_tensor, 
                action = action_tensor,
                model_idxs = model_idxs,
                device = device
                )
    for idx, item in all_mean_mse_losses.items():
        all_mean_mse_losses[idx] = np.array(item)
        print(all_mean_mse_losses[idx].shape)
    for idx, item in all_mean_kl_losses.items():
        all_mean_kl_losses[idx] = np.array(item)
        print(all_mean_kl_losses[idx].shape)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for idx, item in all_mean_mse_losses.items():
        axs[0].plot(item, label = idx)
    axs[0].set_title('Mean MSE loss')
    axs[0].legend()
    for idx, item in all_mean_kl_losses.items():
        axs[1].plot(item, label = idx)
    axs[1].set_title('Mean KL loss')
    axs[1].legend()
    plt.show()




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