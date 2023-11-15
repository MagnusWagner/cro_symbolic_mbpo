from simulation_env.environment_maincrops.environment_maincrops import CropRotationEnv
import numpy as np
import math
from models.utilities.ReplayBufferPrioritized import UniformReplayBuffer, Experience
import torch
import os 
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt

# Write down indices of deterministic, stochastic and static variables


### Input indices:
# 0-23: One-hot encoded actions
# 24: ("N","stochastic"), dependent on everything except prices and costs
# 25: ("P","stochastic"), dependent on everything except prices and costs
# 26: ("K","stochastic"), dependent on everything except prices and costs
# 27: ("Week","stochastic"), dependent on everything except prices and costs
# 28: ("GroundType","static"), static
# 29: ("DryWet","static"), static
# 30: ("Humus","stochastic"), dependent on everything except prices and costs
# 31: ("N_costs","stochastic"), dependent on itself
# 32: ("P_costs","stochastic"), dependent on itself
# 33: ("K_costs","stochastic"), dependent on itself
# 34 - 34+num_crops: ("Prices","stochastic"), dependent on itself
# 34+num_crops:34+2*num_crops ("SowingCosts","stochastic"), dependent on itself 
# 34+2*num_crops:34+3*num_crops ("OtherCosts","stochastic") , dependent on itself 
# 34+3*num_crops:34+8*num_crops ("PreviousCrops","static") ,  static

### Target indices:
# 0: ("Reward","stochastic"), dependent on everything
# 1: ("N","stochastic"), dependent on everything except prices and costs
# 2: ("P","stochastic"), dependent on everything except prices and costs
# 3: ("K","stochastic"), dependent on everything except prices and costs
# 4: ("Week","stochastic"), dependent on everything except prices and costs
# 5: ("GroundType","static"), static
# 6: ("DryWet","static"), static
# 7: ("Humus","stochastic"), dependent on everything except prices and costs
# 8: ("N_costs","stochastic"), dependent on itself
# 9: ("P_costs","stochastic"), dependent on itself
# 10: ("K_costs","stochastic"), dependent on itself
# 11-11+num_crops: ("Prices","stochastic"), dependent on itself
# 11+num_crops:11+2*num_crops ("SowingCosts","stochastic"), dependent on itself 
# 11+2*num_crops:11+3*num_crops ("OtherCosts","stochastic") , dependent on itself 
# 11+3*num_crops:11+8*num_crops ("PreviousCrops","static") ,  static

num_crops = 24
num_input_dimensions = 34+8*num_crops

# generate input indices:
input_action_idxs = list(range(num_crops))
input_condition_idxs = [24,25,26,27,28,29,30] + list(range(34+3*num_crops,34+8*num_crops))
input_prices_and_costs_idxs = [31,32,33] + list(range(34,34+3*num_crops))

# generate target indices:
target_reward_idx = 0
target_stochastic_multi_idxs = [1,2,3,4,7]
target_static_idxs = [5,6] + list(range(11+3*num_crops,11+8*num_crops))
target_prices_and_costs_idxs = [8,9,10] + list(range(11,11+3*num_crops))


# # print lengths of index lists
# print(len(target_stochastic_multi_idxs))
# print(len(target_static_idxs))
# print(len(target_prices_and_costs_idxs))

MODEL_SETTING_DICT = {    
    "stochastic_multi": {
        "type": "stochastic",
        "input_size": len(input_action_idxs + input_condition_idxs),
        "output_size": len(target_stochastic_multi_idxs),
        "num_hidden_units": 933,
        "num_hidden_layers": 3,
        "activation": nn.ReLU(),
        "input_idxs": input_action_idxs + input_condition_idxs,
        "target_idxs": target_stochastic_multi_idxs,
        "lr": 0.00089,
        "weight_decay": 9.18e-08,
        "kl_weight": 0.000135,
        "num_models": 3,
        "stochastic_layer_type": "stochastic_single"
    },
    "reward": {
        "type": "stochastic",
        "input_size": num_input_dimensions,
        "output_size": 1,
        "num_hidden_units": 256,
        "num_hidden_layers": 1,
        "activation": nn.ReLU(),
        "input_idxs": list(range(num_input_dimensions)),
        "target_idxs": [0],
        "lr": 1.53e-02,
        "weight_decay": 2.8e-05,
        "kl_weight": 0.0066,
        "num_models": 3,
        "stochastic_layer_type": "stochastic_single"
    }
}



def get_model_loss_optimizer_pool(seed, device, custom_model_setting_dict = None):
	if custom_model_setting_dict is not None:
		print("Model-Utilities: Custom model setting dict loaded")
		model_setting_dict = custom_model_setting_dict
	else:
		model_setting_dict = MODEL_SETTING_DICT
	torch.manual_seed(seed)
	model_loss_optimizer_pool = {}
	if "reward" in model_setting_dict.keys():
		models = []
		optimizers = []
		for j in range(model_setting_dict["reward"]["num_models"]):
			model, mse_loss, kl_loss, optimizer = create_model_and_losses_and_optimizer(
				input_size = model_setting_dict["reward"]["input_size"],
				output_size = model_setting_dict["reward"]["output_size"],
				num_hidden_units = model_setting_dict["reward"]["num_hidden_units"],
				num_hidden_layers = model_setting_dict["reward"]["num_hidden_layers"],
				stochastic_flag = model_setting_dict["reward"]["stochastic_layer_type"],
				device = device,
				activation = model_setting_dict["reward"]["activation"],
				lr = model_setting_dict["reward"]["lr"],
				weight_decay = model_setting_dict["reward"]["weight_decay"]
				)
			models.append(model)
			optimizers.append(optimizer)
		model_loss_optimizer_pool["reward"] = {
			'models': models,
			'mse_loss': mse_loss, 
			'kl_loss': kl_loss, 
			"optimizers": optimizers, 
			"type": model_setting_dict["reward"]["stochastic_layer_type"],
			"input_idxs": model_setting_dict["reward"]["input_idxs"], 
			"target_idxs": model_setting_dict["reward"]["target_idxs"],
			"kl_weight": model_setting_dict["reward"]["kl_weight"]
			}
	if "stochastic_multi" in model_setting_dict.keys():
		models = []
		optimizers = []       
		for j in range(model_setting_dict["stochastic_multi"]["num_models"]):
			model, mse_loss, kl_loss, optimizer = create_model_and_losses_and_optimizer(
				input_size = model_setting_dict["stochastic_multi"]["input_size"],
				output_size = model_setting_dict["stochastic_multi"]["output_size"],
				num_hidden_units = model_setting_dict["stochastic_multi"]["num_hidden_units"],
				num_hidden_layers = model_setting_dict["stochastic_multi"]["num_hidden_layers"],
				stochastic_flag = model_setting_dict["stochastic_multi"]["stochastic_layer_type"],
				device = device,
				activation = model_setting_dict["stochastic_multi"]["activation"],
				lr = model_setting_dict["stochastic_multi"]["lr"],
				weight_decay = model_setting_dict["stochastic_multi"]["weight_decay"]
			)
			models.append(model)
			optimizers.append(optimizer)
		model_loss_optimizer_pool["stochastic_multi"] = {
			'models': models, 
			'mse_loss': mse_loss,
			'kl_loss': kl_loss, 
			"optimizers": optimizers, 
			"type": model_setting_dict["stochastic_multi"]["stochastic_layer_type"],
			"input_idxs": model_setting_dict["stochastic_multi"]["input_idxs"],
			"target_idxs": model_setting_dict["stochastic_multi"]["target_idxs"],
			"kl_weight": model_setting_dict["stochastic_multi"]["kl_weight"]
		}
	return model_loss_optimizer_pool

def create_model_and_losses_and_optimizer(
		input_size: int, 
		output_size: int, 
		num_hidden_units: int, 
		num_hidden_layers: int, 
		stochastic_flag: bool, 
		device, 
		activation,
		lr: float, 
		weight_decay: float):
    if stochastic_flag == "stochastic_single":
        if num_hidden_layers == 0:
            model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=output_size),    
            )
        else:
            model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=num_hidden_units),
            )
            for i in range(num_hidden_layers-1):
                model.add_module('relu'+str(i), activation)
                model.add_module('hidden'+str(i), nn.Linear(in_features=num_hidden_units, out_features=num_hidden_units))
            model.add_module('relu'+str(num_hidden_layers-1), activation)
            model.add_module('final', nn.Linear(in_features=num_hidden_units, out_features=output_size))
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad = False)
        return model.double().to(device), mse_loss, kl_loss, optimizer
            
    elif stochastic_flag == "none": 
        if num_hidden_layers == 0:
            model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=output_size),    
            )
        else:
            model = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=num_hidden_units),
            )
            for i in range(num_hidden_layers-1):
                model.add_module('relu'+str(i), activation)
                model.add_module('hidden'+str(i), nn.Linear(in_features=num_hidden_units, out_features=num_hidden_units))
            model.add_module('relu'+str(num_hidden_layers-1), activation)
            model.add_module('final', nn.Linear(in_features=num_hidden_units, out_features=output_size))
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad = False)
        return model.double().to(device), mse_loss, optimizer
    elif stochastic_flag == "stochastic_all":
        if num_hidden_layers == 0:
            model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=output_size),    
            )
        else:
            model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_size, out_features=num_hidden_units),
                # nn.Linear(in_features=input_size, out_features=num_hidden_units),
            )
            for i in range(num_hidden_layers-1):
                model.add_module('relu'+str(i), activation)
                model.add_module('hidden'+str(i), bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=num_hidden_units, out_features=num_hidden_units))
            model.add_module('relu'+str(num_hidden_layers-1), activation)
            model.add_module('final', bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=num_hidden_units, out_features=output_size))
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.1
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad = False)
        return model.double().to(device), mse_loss, kl_loss, optimizer








######################################################################
########################## Training functions ########################
######################################################################

# Single Model
####################

def train_single_model_for_one_step(
		inputs, 
		outputs,
		model_key,
		model_idx,
		model_loss_optimizer_pool):
	mse_losses_from_step = []
	kl_losses_from_step = []
	entry = model_loss_optimizer_pool[model_key]
	if "stochastic" in entry["type"]:
		input_idxs = entry["input_idxs"]
		target_idxs = entry["target_idxs"]
		mse_loss_fn = entry["mse_loss"]
		kl_loss_fn = entry["kl_loss"]
		input_filtered = inputs[:,input_idxs]
		target_filtered = outputs[:,target_idxs]
		model = entry["models"][model_idx]
		optimizer = entry["optimizers"][model_idx]
		model.train()
		optimizer.zero_grad()
		output = model(input_filtered)
		mse_loss = mse_loss_fn(output, target_filtered)
		kl_loss = kl_loss_fn(model)
		kl_weight = entry["kl_weight"]
		loss = mse_loss + kl_weight * kl_loss
		loss.backward()
		optimizer.step()
		entry["models"][model_idx] = model 
		entry["optimizers"][model_idx] = optimizer
	return model_loss_optimizer_pool, mse_loss.item(), kl_loss.item()

def train_single_model(
		model_loss_optimizer_pool,
		model_key,
		model_idx, 
		num_steps, 
        replay_buffer,
		batch_size,
        device,
		num_crops):
	mse_losses = []
	kl_losses = []
	for step in range(num_steps):
		idxs, experiences = replay_buffer.uniform_sample(
			replace = True,
			batch_size = batch_size
			)
		states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*experiences))
		inputs, outputs = format_samples_for_training(states, actions, rewards, next_states, device, num_crops)
		model_loss_optimizer_pool, mse_losses_from_step, kl_losses_from_step = train_single_model_for_one_step(
            inputs = inputs, 
            outputs = outputs,
			model_key = model_key,
			model_idx = model_idx,
            model_loss_optimizer_pool = model_loss_optimizer_pool
			)
		mse_losses.append(mse_losses_from_step)
		kl_losses.append(kl_losses_from_step)
        
	mse_losses = np.array(np.log(mse_losses))
	kl_losses = np.array(np.log(kl_losses))
	return model_loss_optimizer_pool, mse_losses, kl_losses


# All Models at the same time
####################
def train_models_for_one_step(
		inputs_list, 
		outputs_list, 
		model_loss_optimizer_pool):
	mse_losses_from_step = []
	kl_losses_from_step = []
	for model_key, entry in model_loss_optimizer_pool.items():
		if "stochastic" in entry["type"]:
			input_idxs = entry["input_idxs"]
			target_idxs = entry["target_idxs"]
			mse_loss_fn = entry["mse_loss"]
			kl_loss_fn = entry["kl_loss"]
			ensemble_mse_losses_from_step = []
			ensemble_kl_losses_from_step = []
			for model_idx in range(len(entry["models"])):
				input_filtered = inputs_list[model_idx][:,input_idxs]
				target_filtered = outputs_list[model_idx][:,target_idxs]
				model = entry["models"][model_idx]
				optimizer = entry["optimizers"][model_idx]
				model.train()
				optimizer.zero_grad()
				output = model(input_filtered)
				mse_loss = mse_loss_fn(output, target_filtered)
				kl_loss = kl_loss_fn(model)
				kl_weight = entry["kl_weight"]
				loss = mse_loss + kl_weight * kl_loss
				loss.backward()
				optimizer.step()
				entry["models"][model_idx] = model 
				entry["optimizers"][model_idx] = optimizer
				ensemble_mse_losses_from_step.append(mse_loss.item())
				ensemble_kl_losses_from_step.append(kl_loss.item())
			mean_ensemble_mse_losses_from_step = np.mean(ensemble_mse_losses_from_step)
			mean_ensemble_kl_losses_from_step = np.mean(ensemble_kl_losses_from_step)
			mse_losses_from_step.append(mean_ensemble_mse_losses_from_step)
			kl_losses_from_step.append(mean_ensemble_kl_losses_from_step)
	return model_loss_optimizer_pool, mse_losses_from_step, kl_losses_from_step

def train_all_models(
		model_loss_optimizer_pool, 
		num_steps, 
        replay_buffer,
		batch_size,
        device,
		num_crops):
	mse_losses = []
	kl_losses = []
	inputs_list = []
	outputs_list = []
	num_models = [len(entry["models"]) for entry in model_loss_optimizer_pool.values()]
	max_num_models = max(num_models)
	for step in range(num_steps):
		for j in range(max_num_models):
			idxs, experiences = replay_buffer.uniform_sample(
				replace = True,
				batch_size = batch_size
				)
			states, actions, rewards, next_states, dones = (torch.stack(vs,0).squeeze(1).to(device) for vs in zip(*experiences))
			inputs, outputs = format_samples_for_training(states, actions, rewards, next_states, device, num_crops)
			inputs_list.append(inputs)
			outputs_list.append(outputs)
		model_loss_optimizer_pool, mse_losses_from_step, kl_losses_from_step = train_models_for_one_step(
            inputs_list = inputs_list, 
            outputs_list = outputs_list,
            model_loss_optimizer_pool = model_loss_optimizer_pool
			)
		mse_losses.append(mse_losses_from_step)
		kl_losses.append(kl_losses_from_step)
        
	mse_losses = np.array(np.log(mse_losses))
	kl_losses = np.array(np.log(kl_losses))
	return model_loss_optimizer_pool, mse_losses, kl_losses

def format_samples_for_training(states, actions, rewards, next_states, device, num_actions):
	delta_states = next_states - states
	# one-hot encode actions
	actions_ohe = torch.nn.functional.one_hot(actions, num_classes=num_actions).squeeze(dim=1)
	inputs = torch.concatenate((actions_ohe, states), axis=-1).to(device).double()
	outputs = torch.concatenate((rewards, delta_states), axis=-1).to(device).double()
	return inputs, outputs

def format_state_action_for_prediction(state, action, num_actions, device):
	# action = torch.tensor(action)
	# one-hot encode actions
	action_ohe = torch.nn.functional.one_hot(action, num_classes=num_actions)
	input = torch.concatenate((action_ohe, state), axis=-1).to(device).double()
	return input

def create_full_replay_buffer(
		length, # filled length of replay buffer 
		seq_len, # sequence-length of the random episodes
		seed, # random seed
		batch_size, # batch size for sampling (unnecessary for uniform sampling)
		device # CUDA or CPU
		):
    env = CropRotationEnv(seed = seed, seq_len = seq_len)
    reward_factor = 5.0/(env.max_reward-env.min_reward)

    random_state = np.random.RandomState(seed)
    replay_buffer = UniformReplayBuffer(batch_size = batch_size,
                                        buffer_size = length,
                                        random_state=random_state)
    # Generate 5 random crop rotations without training (for enviornment testing)
    episodes = math.ceil(length/seq_len)+1
    buffer_length = 0
    while buffer_length < length:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, _, reward, done, _ = env.step(action)
            if done:
                break
            if next_state is None:
                print("next_state is None")

            reward_tensor = torch.tensor([reward], device=device)*reward_factor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            action_tensor = torch.tensor([action]).to(device)
            experience = Experience(state, action_tensor.view(1,1), reward_tensor.view(1,1), next_state_tensor, torch.tensor([done]).view(1,1))
            # print("Stats:", replay_buffer.buffer_size, replay_buffer.batch_size, replay_buffer._buffer_length, replay_buffer._current_idx)
            replay_buffer.add(experience)
            
            state = next_state_tensor
            buffer_length += 1
    return replay_buffer


def create_full_replay_buffer(
		length, # filled length of replay buffer 
		seq_len, # sequence-length of the random episodes
		seed, # random seed
		batch_size, # batch size for sampling (unnecessary for uniform sampling)
		device # CUDA or CPU
		):
    env = CropRotationEnv(seed = seed, seq_len = seq_len)
    reward_factor = 5.0/(env.max_reward-env.min_reward)

    random_state = np.random.RandomState(seed)
    replay_buffer = UniformReplayBuffer(batch_size = batch_size,
                                        buffer_size = length,
                                        random_state=random_state)
    # Generate 5 random crop rotations without training (for enviornment testing)
    episodes = math.ceil(length/seq_len)+1
    buffer_length = 0
    while buffer_length < length:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, _, reward, done, _ = env.step(action)
            if done:
                break
            if next_state is None:
                print("next_state is None")

            reward_tensor = torch.tensor([reward], device=device)*reward_factor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            action_tensor = torch.tensor([action]).to(device)
            experience = Experience(state, action_tensor.view(1,1), reward_tensor.view(1,1), next_state_tensor, torch.tensor([done]).view(1,1))
            # print("Stats:", replay_buffer.buffer_size, replay_buffer.batch_size, replay_buffer._buffer_length, replay_buffer._current_idx)
            replay_buffer.add(experience)
            
            state = next_state_tensor
            buffer_length += 1
    return replay_buffer


######################################################################
########################## Validation plots ##########################
######################################################################
def plot_mse_and_kl_losses(mse_losses, kl_losses):
	print(mse_losses.shape)
	print(kl_losses.shape)
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import seaborn as sns
	sns.set_theme(style="darkgrid")
	plt.figure(figsize=(10,5))
	if len(mse_losses.shape) == 1:
		plt.plot(mse_losses, label = "mse_loss")
	else:
		for i in range(mse_losses.shape[1]):
			plt.plot(mse_losses[:,i], label = "mse_loss_"+str(i))
	plt.legend()
	plt.title("MSE Losses")
	plt.show()
	plt.figure(figsize=(10,5))
	if len(kl_losses.shape) == 1:
		plt.plot(kl_losses, label = "kl_loss")
	else:
		for i in range(kl_losses.shape[1]):
			plt.plot(kl_losses[:,i], label = "kl_loss_"+str(i))
	plt.legend()
	plt.title("KL Losses")
	plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_mse_and_kl_losses_per_key(dynamics_all_mean_mse_losses, dynamics_all_mean_kl_losses):
	sns.set_theme(style="darkgrid")
	# Create two plots besides each other, one for mse_losses and one for kl_losses
	fig, axs = plt.subplots(1, 2, figsize=(20,5))
	for model_key in dynamics_all_mean_mse_losses.keys():
		mse_losses = np.array(dynamics_all_mean_mse_losses[model_key])
		if len(mse_losses.shape) == 1:
			axs[0].plot(mse_losses, label = f"mse_loss {model_key}")
		else:
			for i in range(mse_losses.shape[1]):
				axs[0].plot(mse_losses[:,i], label = f"mse_loss {model_key} {i}")

		axs[0].legend()
	for model_key in dynamics_all_mean_kl_losses.keys():
		kl_losses = np.array(dynamics_all_mean_kl_losses[model_key])
		if len(kl_losses.shape) == 1:
			axs[1].plot(kl_losses, label = f"kl_loss {model_key}")
		else:
			for i in range(kl_losses.shape[1]):
				axs[1].plot(kl_losses[:,i], label = f"kl_loss {model_key} {i}")

		axs[1].legend()
	plt.title("MSE & KL Losses")
	plt.show()
		
     

def plot_validation_fit(model_loss_optimizer_pool, test_inputs, test_outputs, plot_range = 30):
     # Run validation on test_inputs and test_outputs

	# print(model_loss_optimizer_pool[model_key]["input_idxs"])
	# print(model_loss_optimizer_pool[model_key]["target_idxs"])
	for model_key in model_loss_optimizer_pool.keys():
		mse_loss_fn = model_loss_optimizer_pool[model_key]["mse_loss"]
		kl_loss_fn = model_loss_optimizer_pool[model_key]["kl_loss"]
		test_inputs_filtered = test_inputs[:,model_loss_optimizer_pool[model_key]["input_idxs"]]
		test_outputs_filtered = test_outputs[:,model_loss_optimizer_pool[model_key]["target_idxs"]]
		# test_inputs.cpu().shape
		# grid creation:
		n_col = len(model_loss_optimizer_pool[model_key]["models"])
		n_row = test_outputs_filtered.shape[1]
		fig, axs = plt.subplots(n_row, n_col, figsize=(max(8,n_col*4),max(8,n_row*4)))
		for model_idx in range(len(model_loss_optimizer_pool[model_key]["models"])):
			pres = []
			for i in range(30):
				pre = model_loss_optimizer_pool[model_key]["models"][model_idx](test_inputs_filtered)
				pres.append(pre)
			mse = mse_loss_fn(pre, test_outputs_filtered)
			kl = kl_loss_fn(model_loss_optimizer_pool[model_key]["models"][model_idx])
			# # cost = mse + kl_weight*kl
			print('- MSE : %2.6f, KL : %2.6f' % (mse.item(), kl.item()))
			for i in range(test_outputs_filtered.shape[1]):
				if n_row == 1:
					if n_col == 1:
						axs.plot(test_outputs_filtered[:plot_range,i].cpu().detach().numpy(), label='test_outputs', color = (0,0,1))
						for j, pre in enumerate(pres):
							axs.plot(pre[:plot_range,i].cpu().detach().numpy(), label='pre', color = (1,0,0,j/30), alpha = 0.3)
					else:
						axs[model_idx].plot(test_outputs_filtered[:plot_range,i].cpu().detach().numpy(), label='test_outputs', color = (0,0,1))
						for j, pre in enumerate(pres):
							axs[model_idx].plot(pre[:plot_range,i].cpu().detach().numpy(), label='pre', color = (1,0,0,j/30), alpha = 0.3)
				else:
					if n_col == 1:
						axs[i].plot(test_outputs_filtered[:plot_range,i].cpu().detach().numpy(), label='test_outputs', color = (0,0,1))
						for j, pre in enumerate(pres):
							axs[i].plot(pre[:plot_range,i].cpu().detach().numpy(), label='pre', color = (1,0,0,j/30), alpha = 0.3)
					else:
						axs[i][model_idx].plot(test_outputs_filtered[:plot_range,i].cpu().detach().numpy(), label='test_outputs', color = (0,0,1))
						for j, pre in enumerate(pres):
							axs[i][model_idx].plot(pre[:plot_range,i].cpu().detach().numpy(), label='pre', color = (1,0,0,j/30), alpha = 0.3)
				# plt.legend()
		plt.show()


def plot_validation_fit_single_model(
		model_loss_optimizer_pool, 
		test_inputs, 
		test_outputs,
		model_key,
		model_idx,
		plot_range = 30):
     # Run validation on test_inputs and test_outputs

	# print(model_loss_optimizer_pool[model_key]["input_idxs"])
	# print(model_loss_optimizer_pool[model_key]["target_idxs"])

	mse_loss_fn = model_loss_optimizer_pool[model_key]["mse_loss"]
	kl_loss_fn = model_loss_optimizer_pool[model_key]["kl_loss"]
	test_inputs_filtered = test_inputs[:,model_loss_optimizer_pool[model_key]["input_idxs"]]
	test_outputs_filtered = test_outputs[:,model_loss_optimizer_pool[model_key]["target_idxs"]]
	# test_inputs.cpu().shape
	# grid creation:
	n_row = test_outputs_filtered.shape[1]
	pres = []
	for i in range(30):
		pre = model_loss_optimizer_pool[model_key]["models"][model_idx](test_inputs_filtered)
		pres.append(pre)
	mse = mse_loss_fn(pre, test_outputs_filtered)
	kl = kl_loss_fn(model_loss_optimizer_pool[model_key]["models"][model_idx])
	# # cost = mse + kl_weight*kl
	print('- MSE : %2.6f, KL : %2.6f' % (mse.item(), kl.item()))
	for i in range(test_outputs_filtered.shape[1]):
		plt.plot(test_outputs_filtered[:plot_range,i].cpu().detach().numpy(), label='test_outputs', color = (0,0,1))
		for j, pre in enumerate(pres):
			plt.plot(pre[:plot_range,i].cpu().detach().numpy(), label='pre', color = (1,0,0,j/30), alpha = 0.3)
		plt.title(f"Validation plot for model_key {model_key} and model_idx {model_idx}")
		plt.show()

