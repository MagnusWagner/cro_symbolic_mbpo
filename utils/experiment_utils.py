import matplotlib
import matplotlib.pyplot as plt
import torch
import time

def run_experiment(env, agent, steps):
    env.reset()
    history = agent.fit(env, nb_steps=steps, visualize=False, verbose=1)
    length = env.cropRotationSequenceLengthStatic
    # Get average rewards to display in debug output
    reward_list = history.history.get('episode_reward')
    reward_range = 0
    reward_list_average = []
    for i in range(len(reward_list)):
        reward_range += reward_list[i]
        if (i + 1) % length == 0:  # Check if we've completed a rotation sequence
            reward_list_average.append(reward_range / length)
            reward_range = 0
    return reward_list_average

def plot_experiment(reward_list_average):
    iterations = range(0, len(reward_list_average))  # Steps per rotation sequence
    plt.plot(iterations, reward_list_average)
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.title('Average Reward on each step')
    plt.grid(True)
    plt.show()

def plot_losses(losses):
    iterations = range(0, len(losses))  # Steps per rotation sequence
    plt.plot(iterations, losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss on each step')
    plt.grid(True)
    plt.show()


