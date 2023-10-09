from simulation_env.environment_basic import CropRotationEnv
from models.basic.DQNPytorch import DQNAgent
import numpy as np
from utils.experiment_utils import run_experiment, plot_experiment, plot_losses
import torch
from itertools import count

def test_print():
    # Initialize crop rotation environment
    env = CropRotationEnv()

    # Print crops and their attributes for Latex table
    for i in range(len(env.cropNamesDE)):
        print(env.cropNamesEN.get(i) + " & " + str(env.soilNitrogenList.get(i)) + " & " + str(env.cropYieldList.get(i)) + " & " + str(env.cropCultivationBreakList.get(i)) + " & " + str(env.cropMaxCultivationTimesList.get(i)) + " & " + str(env.cropRootCropList.get(i)) + "\\\\")

def test_random():
    env = CropRotationEnv()
    # Generate 5 random crop rotations without training (for enviornment testing)
    episodes = 5
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))



def test_run_pytorch():
    # max_opt_steps = 10
    
    rewards = []
    average_losses = []


    env = CropRotationEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(env)
    num_episodes = 2000     


        
        
    for i_episode in range(num_episodes):
        total_reward = 0
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            losses = []
            action = dqn_agent.select_action(state)
            observation, reward, done, info = env.step(action.item())
            # observation, reward, done, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            dqn_agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            

            losses.append(dqn_agent.optimize_model_onpolicy(state = state, action = action, reward = reward, next_state = next_state))
            dqn_agent.optimize_model_batch()
            state = next_state
            # # Perform one step of the optimization (on the policy network)
            # for i_opt in range(min(i_episode, max_opt_steps)):
            

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = dqn_agent.target_net.state_dict()
            policy_net_state_dict = dqn_agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*dqn_agent.TAU + target_net_state_dict[key]*(1-dqn_agent.TAU)
            dqn_agent.target_net.load_state_dict(target_net_state_dict)

            numeric_reward = reward.item()*(max(env.cropYieldList.values())*1.2-env.negativeReward)
            total_reward += numeric_reward
            if done:
                break
        print(f"Run number: {i_episode}, Reward: {total_reward}, Epsilon: {dqn_agent.eps_threshold}, Average loss: {torch.tensor(losses).mean()}")
        rewards.append(total_reward)
        average_losses.append(torch.tensor(losses).mean())
                
    
    print('Complete')
    plot_experiment(rewards)
    plot_losses(average_losses)




    # reward_list_average = run_experiment(env, dqn_agent, steps)
    # plot_experiment(steps, env.cropRotationSequenceLengthStatic, reward_list_average)

