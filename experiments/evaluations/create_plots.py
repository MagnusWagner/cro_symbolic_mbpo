import matplotlib
import matplotlib.pyplot as plt
import torch
import time

# Create a professional looking plot for an academic paper from a list of rewards, remove unnecessary visual elements except for the main plot, a grid, ticks, a legend, a title and axis descriptions
def plot_experiment(reward_list_average, title, x_label, y_label, grid = True, save_path = None):
    iterations = range(0, len(reward_list_average))  # Steps per rotation sequence
    plt.plot(iterations, reward_list_average)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    if grid:
        plt.grid(True)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

def plot_losses(losses):
    iterations = range(0, len(losses))  # Steps per rotation sequence
    plt.plot(iterations, losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss on each step')
    plt.grid(True)
    plt.show()

def plot_losses_sac(critic1_losses, critic2_losses, actor_losses):
    iterations = range(0, len(critic1_losses))  # Steps per rotation sequence
    plt.plot(iterations, critic1_losses)
    plt.plot(iterations, critic2_losses)
    plt.plot(iterations, actor_losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss on each step')
    plt.legend(["Critic 1", "Critic 2", "Actor"])
    plt.grid(True)
    plt.show()

