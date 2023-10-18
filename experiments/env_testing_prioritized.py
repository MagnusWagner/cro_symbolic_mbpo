from simulation_env.environment_basic import CropRotationEnv
from models.basic.DQN_Prioritized import DeepQAgent
import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from torch import optim
from itertools import count
import collections
import typing
def constant_annealing_schedule(n, constant):
    return constant


def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

def epsilon_decay_schedule(steps_done, EPS_START,EPS_END,EPS_DECAY):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def test_run():
    # max_opt_steps = 10
    
    rewards = []
    average_losses = []


    env = CropRotationEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DeepQAgent(env = env,
                 number_hidden_units = 64,
                 optimizer_fn = lambda parameters: optim.AdamW(parameters, lr=1e-5, amsgrad=False),
                 batch_size = 256,
                 buffer_size = 100000,
                 alpha = 0.7,
                 beta_annealing_schedule = lambda x: exponential_annealing_schedule(x, 5e-3),
                 epsilon_decay_schedule = lambda x: epsilon_decay_schedule(x, 0.9, 0.01, 1200),
                 gamma = 0.99,
                 update_frequency = 1,
                 seed= 42
                 )

    num_episodes = 1500     
   
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            losses = []
            action = dqn_agent.select_action(state)
            observation, reward, done, _ = env.step(action.item())
            
            # observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            
            # Store the transition in memory
            avg_loss = dqn_agent.step(state, action, reward, next_state, done)

            # Move to the next state
            

            losses.append(avg_loss)

            state = next_state
            # # Perform one step of the optimization (on the policy network)
            # for i_opt in range(min(i_episode, max_opt_steps)):
            

            # # Soft update of the target network's weights
            # # θ′ ← τ θ + (1 −τ )θ′
            # target_net_state_dict = dqn_agent.target_net.state_dict()
            # policy_net_state_dict = dqn_agent.policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*dqn_agent.TAU + target_net_state_dict[key]*(1-dqn_agent.TAU)
            # dqn_agent.target_net.load_state_dict(target_net_state_dict)

            numeric_reward = reward.item()*(max(env.cropYieldList.values())*1.2-env.negativeReward)
            total_reward += numeric_reward
            if done:
                break
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Beta: {dqn_agent.beta},  Average loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
                
    
    print('Complete')
    plot_experiment(rewards)
    plot_losses(average_losses)


