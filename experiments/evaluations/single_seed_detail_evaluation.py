from experiments.utilities import single_training_run
from experiments.evaluations.hyperparam_config import hyperparam_config
import numpy as np
import torch 
from torch import optim
from itertools import count
from models.utilities.ReplayBufferPrioritized import UniformReplayBuffer, Experience, Experience_Symbolic
from models.advanced.model_utilities import format_samples_for_training, create_full_replay_buffer, plot_mse_and_kl_losses_per_key, get_filter_informations_from_normalized_states, create_neighbour_replay_buffer
from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv as CropRotationEnv_Advanced
from models.basic.DQN_Prioritized import DeepQAgent as DQN_Prioritized
from models.basic.DQN_Prioritized_Symbolic import DeepQAgent as DQN_Prioritized_Symbolic
from models.advanced.SAC import SACAgent
from models.advanced.SAC_Symbolic import SACAgent as SACAgent_Symbolic
from models.advanced.fake_env import FakeEnv
import typing
import pprint
import copy
from experiments.utilities import single_training_run, run_crop_rotation
import pickle
import time
import clingo

pp = pprint.PrettyPrinter(indent=4)

def single_seed_detail_evaluation():
    SEEDS = [101,102,103,104,105,106,107,108,109,110]
    AGENT_TYPES = ["sac_symbolic"]
    MBRL_KEYS = ["non-mbrl","neighbour"]
    # GROUND_TYPES = [-1.0,0.0,1.0]
    # DRYWETS = [0.0,1.0]
    GROUND_TYPES = [-1.0]
    DRYWETS = [0.0]
    NUM_RUNS = len(SEEDS)*len(AGENT_TYPES)*len(MBRL_KEYS)*len(GROUND_TYPES)*len(DRYWETS)
    print("Number of runs:",NUM_RUNS)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENVIRONMENT_TYPE = "advanced"
    SEQ_LEN = 10
    RULE_OPTIONS = "humus_and_breaks"
    PRETRAIN_BUFFER_SIZE = 5000
    PRETRAIN_NUM_STEPS = 500
    NUM_NEIGHBOURS = 50
    NEIGHBOUR_BUFFER_SIZE = 5000
    # NUM_EPISODES = 200
    NUM_EPISODES = 200
    TRAINING_EVAL_RATIO = 10
    start_time = time.time()
    i_run = 1
    best_episode_total_reward = -50000
    best_episode_index = None
    best_episode_crops = None
    best_episode_rewards = None
    for seed in SEEDS:
        for ground_type in GROUND_TYPES:
            for drywet in DRYWETS:
                
                # Pretraining FakeEnv for mbrl training
                # random_state1 = np.random.RandomState(seed)
                # fake_env = FakeEnv(device = DEVICE, random_state = random_state1)
                # print("Pretrain dynamics model.")

                # pretrain_replay_buffer = create_full_replay_buffer(length = PRETRAIN_BUFFER_SIZE, seq_len=SEQ_LEN, random_state = random_state1, DryWetInit = drywet, GroundTypeInit = ground_type, batch_size = 128, device = DEVICE, filter_flag = True, rule_options = RULE_OPTIONS)
                # for i in range(PRETRAIN_NUM_STEPS):
                #     if i % 100 == 0:
                #         print(f"Pretrain-Dynamics-Model-Episode {i}/{PRETRAIN_NUM_STEPS}")
                #     fake_env.train(
                #         replay_buffer = pretrain_replay_buffer, 
                #         num_steps = 10, 
                #         batch_size = 256,
                #     )
                # print("Pretraining of dynamics model finished.")

                # Neighbour experiment
                print("Neighbour experiment")
                neighbour_replay_buffers = {}

                # Create symbolic replay buffer for neighbour experiment
                random_state_env = np.random.RandomState(seed)
                env = CropRotationEnv_Advanced(seq_len=10, random_state = random_state_env)
                neighbour_replay_buffers["symbolic"] = create_neighbour_replay_buffer(
                    env = env, 
                    num_neighbours = NUM_NEIGHBOURS, 
                    length = NEIGHBOUR_BUFFER_SIZE, 
                    seq_len = SEQ_LEN, 
                    random_state = random_state_env,
                    batch_size = 256,
                    device = DEVICE, 
                    neighbour_alpha = 1.0,
                    filter_flag = True, 
                    rule_options = RULE_OPTIONS)

                # Create non-symbolic replay buffer for neighbour experiment
                random_state_env = np.random.RandomState(seed)
                env = CropRotationEnv_Advanced(seq_len=10, random_state = random_state_env)
                neighbour_replay_buffers["non-symbolic"] = create_neighbour_replay_buffer(
                    env = env, 
                    num_neighbours = NUM_NEIGHBOURS, 
                    length = NEIGHBOUR_BUFFER_SIZE, 
                    seq_len = SEQ_LEN, 
                    random_state = random_state_env,
                    batch_size = 256,
                    device = DEVICE, 
                    neighbour_alpha = 1.0, 
                    filter_flag = False, 
                    rule_options = RULE_OPTIONS)

                for agent_type in AGENT_TYPES:
                    symbolic_key = "symbolic" if "symbolic" in agent_type else "non-symbolic"
                    agent_type_key = "sac" if "sac" in agent_type else "prioritized"
                    for mbrl_key in MBRL_KEYS:
                        param_dict = hyperparam_config[ENVIRONMENT_TYPE][agent_type_key][symbolic_key][mbrl_key]

                        print(f"Current-Run: {i_run}/{NUM_RUNS}.")
                        print(f"Current time taken: {round((time.time() - start_time)/60,2)} minutes.") 
                        print(f"Attributes - Agent: {agent_type}, MBRL-Key: {mbrl_key}, Ground-Type: {ground_type}, DryWet: {drywet}, Seed: {seed}")
                        i_run += 1
                        results, _ = single_training_run(
                            param_dict,
                            agent_type = agent_type,
                            environment_type = ENVIRONMENT_TYPE,
                            num_episodes = NUM_EPISODES,
                            training_eval_ratio = TRAINING_EVAL_RATIO,
                            # Environment parameters
                            DryWetInit = drywet,
                            GroundTypeInit = ground_type,
                            deterministic = True,
                            seq_len = SEQ_LEN,
                            # Symbolic parameters
                            only_filter = False,
                            rule_options = RULE_OPTIONS,
                            # MBRL parameters
                            mbrl_flag = True if mbrl_key == "mbrl" else False,
                            pretrain_flag = True if mbrl_key == "mbrl" else False,
                            pretrain_buffer_size = PRETRAIN_BUFFER_SIZE,
                            pretrain_num_steps = PRETRAIN_NUM_STEPS,
                            pretrained_fake_env = None,
                            # pretrained_fake_env = fake_env if mbrl_key == "mbrl" else None,
                            # Neighbour parameters
                            neighbour_flag = True if mbrl_key == "neighbour" else False,
                            num_neighbours = NUM_NEIGHBOURS,
                            neighbour_buffer_size = NEIGHBOUR_BUFFER_SIZE,
                            neighbourhood_replay_buffer = neighbour_replay_buffers[symbolic_key] if mbrl_key == "neighbour" else None,
                            plot_flag = False,
                            print_flag = False,
                            detailed_tracking_flag=True,
                            seed = seed)
                        file_path = f"experiments/evaluations/results/detailed_{agent_type}_{mbrl_key}_{ground_type}_{drywet}_{seed}.pickle"
                        with open(file_path, 'wb') as handle:
                            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        max_episode_reward = np.max(results["training_rewards"])
                        if max_episode_reward > best_episode_total_reward:
                            best_episode_index = np.argmax(results["training_rewards"])
                            best_episode_crops = results["crops_selected_idxs_list"][best_episode_index]
                            best_episode_rewards = results["rewards_list_list"][best_episode_index]
                            best_episode_total_reward = max_episode_reward
                print("Best episode crops:",best_episode_crops)
                print("Best episode rewards:",best_episode_rewards)
                print("Best episode total reward:",best_episode_total_reward)

                # run crop rotation for environment
                results = run_crop_rotation(
                    environment_type = ENVIRONMENT_TYPE,
                    num_episodes = NUM_EPISODES,
                    training_eval_ratio = TRAINING_EVAL_RATIO,
                    # Environment parameters
                    DryWetInit = drywet,
                    GroundTypeInit = ground_type,
                    deterministic = True,
                    seq_len = SEQ_LEN,
                    crop_idxs = best_episode_crops,
                    detailed_tracking_flag=True,
                    seed = seed)
                results["used_crop_rotation"] = best_episode_crops
                file_path = f"experiments/evaluations/results/detailed_fixed_croprota_{ground_type}_{drywet}_{seed}.pickle"
                with open(file_path, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print("Pipeline finished.")
    print(f"Total time taken: {round((time.time() - start_time)/60,2)} minutes.") 
    # Only-Filter run
    param_dict = hyperparam_config[ENVIRONMENT_TYPE]["prioritized"]["symbolic"]["non-mbrl"]

    for seed in SEEDS:
        for ground_type in GROUND_TYPES:
            for drywet in DRYWETS:
                results, _ = single_training_run(
                    param_dict,
                    agent_type = "prioritized_symbolic",
                    environment_type = ENVIRONMENT_TYPE,
                    num_episodes = NUM_EPISODES,
                    training_eval_ratio = TRAINING_EVAL_RATIO,
                    # Environment parameters
                    DryWetInit = drywet,
                    GroundTypeInit = ground_type,
                    deterministic = True,
                    seq_len = SEQ_LEN,
                    # Symbolic parameters
                    only_filter = True,
                    rule_options = RULE_OPTIONS,
                    # MBRL parameters
                    mbrl_flag = False,
                    pretrain_flag = False,
                    pretrain_buffer_size = PRETRAIN_BUFFER_SIZE,
                    pretrain_num_steps = PRETRAIN_NUM_STEPS,
                    pretrained_fake_env = None,
                    # Neighbour parameters
                    neighbour_flag = False,
                    num_neighbours = NUM_NEIGHBOURS,
                    neighbour_buffer_size = NEIGHBOUR_BUFFER_SIZE,
                    neighbourhood_replay_buffer = None,
                    plot_flag = False,
                    seed = seed)
                file_path = f"experiments/evaluations/results/detailed_symbolic_only_filter_{ground_type}_{drywet}_{seed}.pickle"
                with open(file_path, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Random Run
    param_dict = hyperparam_config[ENVIRONMENT_TYPE]["prioritized"]["non-symbolic"]["non-mbrl"]

    for seed in SEEDS:
        for ground_type in GROUND_TYPES:
            for drywet in DRYWETS:
                results, _ = single_training_run(
                    param_dict,
                    agent_type = "prioritized",
                    environment_type = ENVIRONMENT_TYPE,
                    num_episodes = NUM_EPISODES,
                    training_eval_ratio = TRAINING_EVAL_RATIO,
                    # Environment parameters
                    DryWetInit = drywet,
                    GroundTypeInit = ground_type,
                    deterministic = True,
                    seq_len = SEQ_LEN,
                    # Symbolic parameters
                    only_filter = False,
                    rule_options = RULE_OPTIONS,
                    random_flag = True,
                    # MBRL parameters
                    mbrl_flag = False,
                    pretrain_flag = False,
                    pretrain_buffer_size = PRETRAIN_BUFFER_SIZE,
                    pretrain_num_steps = PRETRAIN_NUM_STEPS,
                    pretrained_fake_env = None,
                    # Neighbour parameters
                    neighbour_flag = False,
                    num_neighbours = NUM_NEIGHBOURS,
                    neighbour_buffer_size = NEIGHBOUR_BUFFER_SIZE,
                    neighbourhood_replay_buffer = None,
                    plot_flag = False,
                    seed = seed)
                file_path = f"experiments/evaluations/results/detailed_only_random_{ground_type}_{drywet}_{seed}.pickle"
                with open(file_path, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)



def run_fixed_crop_rotation(ground_type, drywet, crop_idxs):

    SEEDS = [101,102,103,104,105,106,107,108,109,110]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENVIRONMENT_TYPE = "advanced"
    SEQ_LEN = 10
    RULE_OPTIONS = "humus_and_breaks"
    PRETRAIN_BUFFER_SIZE = 5000
    PRETRAIN_NUM_STEPS = 500
    NUM_NEIGHBOURS = 50
    NEIGHBOUR_BUFFER_SIZE = 5000
    # NUM_EPISODES = 200
    NUM_EPISODES = 200
    TRAINING_EVAL_RATIO = 10
    start_time = time.time()
    i_run = 1
    for seed in SEEDS:
        # run crop rotation for environment
        results = run_crop_rotation(
            environment_type = ENVIRONMENT_TYPE,
            num_episodes = NUM_EPISODES,
            training_eval_ratio = TRAINING_EVAL_RATIO,
            # Environment parameters
            DryWetInit = drywet,
            GroundTypeInit = ground_type,
            deterministic = True,
            seq_len = SEQ_LEN,
            crop_idxs = crop_idxs,
            detailed_tracking_flag=True,
            seed = seed)
        file_path = f"experiments/evaluations/results/detailed_foreign_croprota_{ground_type}_{drywet}_{seed}.pickle"
        with open(file_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print("Pipeline finished.")
    print(f"Total time taken: {round((time.time() - start_time)/60,2)} minutes.") 