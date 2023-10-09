from simulation_env.environment_basic import CropRotationEnv
from models.original import DQNKeras
import keras 
from utils.experiment_utils import run_experiment, plot_experiment
import torch

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

def test_run():
    env = CropRotationEnv()
    dqn = DQNKeras(env)
    dqn_agent = dqn.build_agent()
    dqn_agent.compile(keras.optimizers.Adam(learning_rate=0.035), metrics=['mae']) #war 0.035
    steps = 15000
    # Example usage:
    # Assuming you have already run your experiment and obtained reward_list_average
    reward_list_average = run_experiment(env, dqn_agent, steps)
    plot_experiment(steps, env.cropRotationSequenceLengthStatic, reward_list_average)

