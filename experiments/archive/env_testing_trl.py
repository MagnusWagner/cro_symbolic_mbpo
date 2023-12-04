from simulation_env.environment_basic import CropRotationEnv
from models.basic.DQNPytorch import DQNAgent

from utils.experiment_utils import run_experiment, plot_experiment
import torch
from itertools import count

##################
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
##################



# def test_print():
#     # Initialize crop rotation environment
#     env = CropRotationEnv()

#     # Print crops and their attributes for Latex table
#     for i in range(len(env.cropNamesDE)):
#         print(env.cropNamesEN.get(i) + " & " + str(env.soilNitrogenList.get(i)) + " & " + str(env.cropYieldList.get(i)) + " & " + str(env.cropCultivationBreakList.get(i)) + " & " + str(env.cropMaxCultivationTimesList.get(i)) + " & " + str(env.cropRootCropList.get(i)) + "\\\\")

# def test_random():
#     env = CropRotationEnv()
#     # Generate 5 random crop rotations without training (for enviornment testing)
#     episodes = 5
#     for episode in range(1, episodes+1):
#         state = env.reset()
#         done = False
#         score = 0 
        
#         while not done:
#             action = env.action_space.sample()
#             n_state, reward, done, info = env.step(action)
#             score+=reward
#         print('Episode:{} Score:{}'.format(episode, score))



def test_run_pytorch():
    device = "cpu" if not torch.has_cuda else "cuda:0"
    num_cells = 32  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimisation steps per batch of data collected
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4
    rewards = []


    env = CropRotationEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(env)

    if torch.cuda.is_available():
        num_episodes = 1000
    else:
        num_episodes = 1000

        
        
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = dqn_agent.select_action(state)
            observation, reward, done, info = env.step(action)
            # observation, reward, done, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            dqn_agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            

            dqn_agent.optimize_model_onpolicy(state = state, action = action, reward = reward, next_state = next_state)
            state = next_state
            # Perform one step of the optimization (on the policy network)
            for i_opt in range(min(i_episode, max_opt_steps)):
                dqn_agent.optimize_model_single()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = dqn_agent.target_net.state_dict()
            policy_net_state_dict = dqn_agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*dqn_agent.TAU + target_net_state_dict[key]*(1-dqn_agent.TAU)
            dqn_agent.target_net.load_state_dict(target_net_state_dict)

            numeric_reward = reward.item()
            total_reward += numeric_reward
            if done:
                break
        print(f"#{i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}")
        rewards.append(total_reward)
                
    
    print('Complete')
    plot_experiment(num_episodes, env.cropRotationSequenceLengthStatic,rewards)




    # reward_list_average = run_experiment(env, dqn_agent, steps)
    # plot_experiment(steps, env.cropRotationSequenceLengthStatic, reward_list_average)

