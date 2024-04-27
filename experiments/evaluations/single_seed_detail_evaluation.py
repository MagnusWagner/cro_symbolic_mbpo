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
    # SEEDS = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120]
    # SEEDS = 
    SEEDS = [102]
    AGENT_TYPES = ["sac_symbolic"]
    MBRL_KEYS = ["neighbour"]
    # GROUND_TYPES = [-1.0,0.0,1.0]
    # DRYWETS = [0.0,1.0]
    GROUND_TYPES = [0.0]
    DRYWETS = [1.0]
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
    NUM_EPISODES = 200
    # NUM_EPISODES = 500
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



def run_fixed_crop_rotation(ground_type, drywet, crop_idxs_init = None):
    SEEDS = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120]
    
    
    sequence_dicts = {   5: {   101: [18, 20, 11, 5, 23, 14, 2, 17, 4, 13],
           102: [14, 6, 0, 17, 9, 19, 8, 5, 16],
           103: [21, 16, 0, 11, 20, 17, 12, 16, 11],
           104: [0, 22, 2, 13, 11, 8, 18, 8, 2],
           105: [9, 12, 5, 21, 23, 19, 17, 19, 23, 22],
           106: [0, 4, 8, 18, 6, 11, 19, 12, 19],
           107: [8, 12, 8, 0, 0, 10, 18, 6],
           108: [11, 12, 2, 19, 19, 23, 21, 4, 11, 18],
           109: [21, 3, 14, 10, 0, 9, 19, 18, 23],
           110: [18, 23, 12, 19, 8, 2, 20, 0, 9],
           111: [21, 13, 4, 4, 12, 23, 17, 8, 18, 23],
           112: [12, 5, 20, 9, 23, 3, 8, 5, 17, 18],
           113: [20, 18, 14, 2, 23, 21, 16, 13, 19, 23],
           114: [6, 2, 20, 11, 6, 23, 19, 12, 0],
           115: [18, 19, 23, 3, 1, 4, 2, 11, 18],
           116: [4, 12, 20, 9, 23, 3, 14, 19, 8, 20],
           117: [5, 13, 18, 23, 6, 18, 11, 14, 19, 13],
           118: [5, 6, 12, 14, 22, 0, 20, 15, 6],
           119: [15, 4, 3, 3, 14, 4, 12, 8, 21, 18],
           120: [1, 11, 19, 13, 21, 9, 5, 8, 19]},
    10: {   101: [18, 20, 11, 5, 23, 14, 2, 17, 4, 13],
            102: [20, 17, 4, 20, 8, 23, 19, 18, 6, 4],
            103: [21, 11, 19, 19, 18, 23, 7, 19, 23, 4],
            104: [7, 1, 18, 17, 19, 2, 20, 10, 18],
            105: [4, 18, 23, 19, 19, 20, 22, 2, 23, 19],
            106: [0, 4, 8, 18, 6, 11, 19, 12, 19],
            107: [12, 15, 5, 16, 14, 23, 19, 18, 23, 21],
            108: [20, 13, 18, 23, 19, 19, 23, 16, 21, 19],
            109: [1, 9, 21, 9, 19, 18, 23, 19, 19],
            110: [18, 23, 12, 19, 8, 2, 20, 0, 9],
            111: [21, 13, 4, 4, 12, 23, 17, 8, 18, 23],
            112: [4, 6, 4, 19, 23, 1, 18, 14, 4],
            113: [22, 2, 23, 0, 15, 12, 18, 17, 2],
            114: [6, 2, 20, 11, 6, 23, 19, 12, 0],
            115: [18, 19, 23, 3, 1, 4, 2, 11, 18],
            116: [20, 19, 23, 21, 16, 5, 13, 11, 20, 19],
            117: [8, 13, 2, 18, 3, 23, 19, 13, 20, 4],
            118: [8, 19, 23, 13, 2, 18, 15, 18, 12, 4],
            119: [10, 19, 23, 21, 6, 1, 21, 5, 22],
            120: [1, 11, 19, 13, 21, 9, 5, 8, 19]},
    20: {   101: [18, 20, 11, 5, 23, 14, 2, 17, 4, 13],
            102: [0, 10, 5, 17, 19, 7, 19, 12, 18],
            103: [20, 4, 10, 12, 18, 23, 19, 23, 18, 19],
            104: [7, 13, 18, 23, 19, 23, 4, 18, 19, 18],
            105: [3, 10, 6, 8, 19, 5, 23, 21, 19, 19],
            106: [12, 7, 5, 21, 23, 18, 19, 12, 4, 20],
            107: [19, 23, 21, 17, 17, 19, 17, 5, 19, 23],
            108: [18, 23, 17, 8, 19, 17, 19, 23, 5, 19],
            109: [1, 9, 21, 9, 19, 18, 23, 19, 19],
            110: [19, 23, 19, 13, 17, 19, 8, 3, 19, 18],
            111: [4, 18, 23, 9, 19, 19, 11, 5, 23, 19],
            112: [18, 19, 16, 18, 23, 18, 23, 19, 8, 18],
            113: [14, 1, 21, 4, 17, 18, 21, 19, 16],
            114: [19, 6, 19, 19, 23, 5, 19, 23, 19, 18],
            115: [19, 21, 4, 18, 23, 19, 19, 17, 5, 19],
            116: [0, 10, 21, 19, 17, 5, 19, 23, 17],
            117: [8, 13, 2, 18, 3, 23, 19, 13, 20, 4],
            118: [18, 11, 13, 4, 20, 23, 19, 18, 23, 19],
            119: [5, 4, 13, 19, 4, 21, 19, 23, 16, 12],
            120: [1, 11, 19, 13, 21, 9, 5, 8, 19]},
    50: {   101: [21, 11, 4, 10, 23, 19, 18, 23, 19, 19],
            102: [0, 10, 5, 17, 19, 7, 19, 12, 18],
            103: [20, 4, 10, 12, 18, 23, 19, 23, 18, 19],
            104: [18, 23, 19, 18, 10, 19, 23, 18, 12, 19],
            105: [5, 18, 23, 19, 8, 19, 18, 19, 8, 4],
            106: [21, 8, 4, 12, 23, 19, 17, 12, 21, 19],
            107: [19, 23, 21, 17, 17, 19, 17, 5, 19, 23],
            108: [8, 12, 18, 23, 19, 23, 19, 19, 18, 21],
            109: [18, 19, 23, 4, 19, 23, 18, 19, 19, 23],
            110: [19, 23, 19, 13, 17, 19, 8, 3, 19, 18],
            111: [4, 18, 23, 9, 19, 19, 11, 5, 23, 19],
            112: [18, 19, 16, 18, 23, 18, 23, 19, 8, 18],
            113: [20, 4, 19, 23, 18, 17, 0, 18, 18],
            114: [19, 6, 19, 19, 23, 5, 19, 23, 19, 18],
            115: [18, 23, 19, 15, 7, 21, 19, 23, 19, 18],
            116: [4, 19, 18, 23, 19, 23, 19, 17, 8, 4],
            117: [8, 13, 2, 18, 3, 23, 19, 13, 20, 4],
            118: [9, 12, 12, 18, 19, 23, 4, 15, 19, 23],
            119: [18, 23, 13, 19, 8, 22, 5, 23, 19, 18],
            120: [2, 12, 19, 23, 17, 19, 23, 20, 17, 19]},
    100: {   101: [4, 17, 23, 19, 18, 19, 23, 18, 17, 19],
             102: [19, 23, 18, 17, 19, 18, 23, 19, 23, 18],
             103: [20, 4, 10, 12, 18, 23, 19, 23, 18, 19],
             104: [18, 23, 19, 18, 10, 19, 23, 18, 12, 19],
             105: [0, 19, 18, 17, 19, 18, 23, 19, 23],
             106: [12, 19, 23, 19, 18, 23, 17, 19, 18, 23],
             107: [14, 13, 4, 19, 23, 17, 19, 23, 18, 19],
             108: [8, 12, 18, 23, 19, 23, 19, 19, 18, 21],
             109: [18, 19, 23, 4, 19, 23, 18, 19, 19, 23],
             110: [19, 23, 19, 13, 17, 19, 8, 3, 19, 18],
             111: [4, 18, 23, 9, 19, 19, 11, 5, 23, 19],
             112: [19, 23, 19, 14, 4, 17, 19, 12, 23, 19],
             113: [19, 18, 23, 19, 8, 4, 18, 23, 13, 19],
             114: [19, 6, 19, 19, 23, 5, 19, 23, 19, 18],
             115: [4, 17, 23, 19, 18, 9, 19, 18, 19, 23],
             116: [19, 23, 18, 17, 19, 23, 18, 10, 4, 9],
             117: [12, 19, 23, 17, 13, 8, 18, 23, 19, 19],
             118: [19, 23, 4, 18, 17, 14, 19, 17, 5, 18],
             119: [18, 21, 5, 23, 19, 11, 18, 23, 19, 18],
             120: [19, 19, 23, 17, 19, 23, 18, 17, 19, 17]},
    500: {   101: [4, 17, 19, 23, 9, 4, 8, 18, 23, 19],
             102: [8, 4, 19, 23, 18, 17, 4, 19, 23, 18],
             103: [19, 23, 18, 17, 12, 19, 18, 23, 5, 19],
             104: [18, 23, 19, 18, 10, 19, 23, 18, 12, 19],
             105: [19, 23, 18, 19, 23, 17, 19, 12, 18, 19],
             106: [6, 12, 19, 18, 23, 19, 23, 18, 12, 19],
             107: [7, 4, 19, 23, 18, 19, 23, 19, 18, 12],
             108: [19, 23, 18, 18, 17, 19, 18, 23, 19, 18],
             109: [19, 23, 18, 19, 23, 17, 4, 18, 19, 18],
             110: [19, 18, 18, 23, 19, 23, 18, 19, 8, 4],
             111: [19, 23, 18, 19, 23, 17, 8, 19, 23, 19],
             112: [19, 18, 23, 9, 19, 23, 18, 23, 19, 18],
             113: [8, 18, 19, 23, 18, 23, 19, 8, 12, 18],
             114: [19, 23, 19, 17, 19, 23, 19, 18, 19, 23],
             115: [4, 17, 23, 19, 18, 17, 5, 23, 19, 18],
             116: [4, 19, 23, 18, 17, 19, 23, 18, 19, 18],
             117: [19, 23, 18, 17, 19, 23, 17, 19, 18, 17],
             118: [8, 19, 23, 18, 9, 1, 17, 19, 18],
             119: [19, 18, 23, 19, 23, 18, 17, 19, 23, 18],
             120: [0, 17, 19, 18, 19, 23, 19, 18, 19]}}

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENVIRONMENT_TYPE = "advanced"
    SEQ_LEN = 10
    RULE_OPTIONS = "humus_and_breaks"
    PRETRAIN_BUFFER_SIZE = 5000
    PRETRAIN_NUM_STEPS = 500
    NUM_NEIGHBOURS = 50
    NEIGHBOUR_BUFFER_SIZE = 5000
    NUM_EPISODES = 200
    # NUM_EPISODES = 500
    TRAINING_EVAL_RATIO = 10
    start_time = time.time()
    i_run = 1
    for seed in SEEDS:
        for r, sequence_dict in sequence_dicts.items():
            crop_idxs = sequence_dict[seed]
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
            file_path = f"experiments/evaluations/results/detailed_best_croprota_{ground_type}_{drywet}_{seed}_range_{r}.pickle"
            with open(file_path, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print("Pipeline finished.")
    print(f"Total time taken: {round((time.time() - start_time)/60,2)} minutes.") 


