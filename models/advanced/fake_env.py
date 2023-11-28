# SET LENGTH HERE
import numpy as np
from simulation_env.environment_maincrops.data.mappings import crop_mapping_german, crop_mapping_german_rev, crop_mapping_eng, date_mapping, date_mapping_rev
from simulation_env.environment_maincrops.data.cropbreaks import cropbreaks, mf_groups, ap_groups
from simulation_env.environment_maincrops.data.kolbe import kolbe_matrix
import json
import os
import pprint
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
from models.advanced.model_utilities import get_model_loss_optimizer_pool, train_all_models, format_samples_for_training, plot_mse_and_kl_losses, plot_validation_fit, format_state_action_for_prediction, train_single_model
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)
def convert_state_to_index(state):
   return np.argmax(state)

def convert_index_to_state(length, index):
  state = np.zeros(length)
  state[index] = 1.0
  return state

print(os.getcwd())
json_file_path = "./simulation_env/environment_maincrops/data/maincrops_updated.json"
# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    # Parse the JSON data into a Python dictionary
    maincrop_yields = json.load(json_file)

json_file_path = "./simulation_env/environment_maincrops/data/maincrop_properties.json"
# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    # Parse the JSON data into a Python dictionary
    maincrop_properties = json.load(json_file)

class FakeEnv:

    def __init__(self, device, random_state, custom_model_setting_dict = None):
        self.device = device
        self.random_state = random_state

        # Maincrop yields
        self.maincrop_yields = maincrop_yields
        self.maincrop_properties = maincrop_properties

        # German names of crops used in the crop rotation
        self.crop_mapping_german = crop_mapping_german
        self.crop_mapping_german_rev = crop_mapping_german_rev
        
        # English names of crops used in the crop rotation
        self.crop_mapping_eng = crop_mapping_eng

        # Number of crops
        self.num_crops = len(self.crop_mapping_german)

        # Mapping dates to integer indices
        self.date_mapping = date_mapping
        self.date_mapping_rev = date_mapping_rev

        # Target Indices for model
        self.target_stochastic_multi_idxs = np.array([0,1,2,3,6])
        self.target_static_idxs = np.array([4,5])
        self.target_previous_crops_idxs = np.array(list(range(10+3*self.num_crops,10+8*self.num_crops)))
        self.target_gbm_idxs = np.array(list(range(7,10+3*self.num_crops)))


        self.GroundType = None
        self.DryWet = None
        
        self.previous_state = None
        self.state = None
        self.state_normalized = None
        # Action and observation space are defined by crops in cropNames
        # State size = 1 (0: Nitrogen-Level) + 1 (1: Phosphor-Level) + 1 (2: Kalium-Level) + 1 (3: Week index of year) + 1 (4: Ground type) + 1 (5: Dry/Wet year) + 1 (6: Humus %) + 3 (7-9: fertilizer costs) \n
        # 3*num_crops (prices[10:10+num_crops] + sowing[10+num_crops:10+num_crops*2] + other costs[10+num_crops*2:10+num_crops*3]) + 5*num_crops (previous crops selected) = 192
        self.state_size = 10 + 3*self.num_crops + 5*self.num_crops
        self.input_size = self.state_size + self.num_crops
        self.output_size = self.state_size + 1
        # Previous crops initialization
        self.previous_crop = None
        self.previous_crops_selected = np.array([None,None,None,None,None])
        self.previous_crops_selected_matrix = np.zeros((5, self.num_crops))

        
        # GBM & average stats for prices and costs
        self.prices_gbm_avg = torch.tensor([crop["Verkaufspreis"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.prices_gbm_std = torch.tensor([crop["Verkaufspreis"]["std_gbm"] for crop in self.maincrop_yields.values()])

        self.sowing_costs_gbm_avg = torch.tensor([crop["Kosten"]["Saatgut"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.sowing_costs_gbm_std = torch.tensor([crop["Kosten"]["Saatgut"]["std_gbm"] for crop in self.maincrop_yields.values()])

        self.other_costs_gbm_avg = torch.tensor([crop["Kosten"]["Sonstiges"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.other_costs_gbm_std = torch.tensor([crop["Kosten"]["Sonstiges"]["std_gbm"] for crop in self.maincrop_yields.values()])
        
        self.N_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["avg_gbm"]
        self.N_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["std_gbm"]

        self.P_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["avg_gbm"]
        self.P_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["std_gbm"]

        self.K_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["avg_gbm"]
        self.K_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["std_gbm"]

        self.maximum_prices = torch.tensor([crop["Verkaufspreis"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_sowing_costs = torch.tensor([crop["Kosten"]["Saatgut"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_other_costs = torch.tensor([crop["Kosten"]["Sonstiges"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_N_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["max"]*2
        self.maximum_P_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["max"]*2
        self.maximum_K_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["max"]*2

        self.gbm_avg = torch.concatenate((torch.tensor([self.N_costs_gbm_avg, self.P_costs_gbm_avg, self.K_costs_gbm_avg]),self.prices_gbm_avg, self.sowing_costs_gbm_avg, self.other_costs_gbm_avg)).to(self.device)
        self.gbm_std = torch.concatenate((torch.tensor([self.N_costs_gbm_std, self.P_costs_gbm_std, self.K_costs_gbm_std]),self.prices_gbm_std, self.sowing_costs_gbm_std, self.other_costs_gbm_std)).to(self.device)
        self.gbm_max = torch.concatenate((torch.tensor([self.maximum_N_costs, self.maximum_P_costs, self.maximum_K_costs]),self.maximum_prices, self.maximum_sowing_costs, self.maximum_other_costs)).to(self.device)
        # The previous crop selection is one-hot-encoded and therefore stays between 0 and 1
        self.model_loss_optimizer_pool = get_model_loss_optimizer_pool(
             random_state = self.random_state, 
             device = self.device, 
             custom_model_setting_dict=custom_model_setting_dict)

    def get_num_models(self):
        num_models_list = [len(self.model_loss_optimizer_pool[key]["models"]) for key in self.model_loss_optimizer_pool.keys()]
        return num_models_list
    
    def get_model_keys(self):
        return self.model_loss_optimizer_pool.keys()

    def calculate_next_gbms(self, last_values):
        last_values = last_values * self.gbm_max
        assert last_values.shape == self.gbm_avg.shape == self.gbm_std.shape
        return torch.minimum(self.gbm_max,last_values * torch.exp((self.gbm_avg - (self.gbm_std ** 2) / 2) + self.gbm_std * self.random_state.normal()))/self.gbm_max

    def get_next_state_from_prediction(self, state, predicted_stochastic_multi, action):
        # state = torch.clone(state[0])
        state = state.squeeze()
        next_state = torch.empty(self.state_size).to(self.device)
        next_state[self.target_stochastic_multi_idxs] = state[self.target_stochastic_multi_idxs]
        next_state[self.target_stochastic_multi_idxs] += predicted_stochastic_multi.float()
        next_state[self.target_static_idxs] = state[self.target_static_idxs]
        next_state[self.target_previous_crops_idxs] = self.get_next_previous_crops_array(state[self.target_previous_crops_idxs], action)
        next_state[self.target_gbm_idxs] = self.calculate_next_gbms(state[self.target_gbm_idxs])
        # assert that no nan values are in the next state
        assert not torch.isnan(next_state).any()
        return next_state.unsqueeze(0)

    def get_next_previous_crops_array(self, previous_crops_selected_matrix, action):
        action = action.item()
        # reshape previous crops array to 2d array
        previous_crops_selected_matrix = previous_crops_selected_matrix.reshape((5, self.num_crops))
        if action in [0,1]:
            current_crop_selection_vector = torch.zeros((2,self.num_crops)).to(self.device)
            current_crop_selection_vector[:,action] = 1
            previous_crops_selected_matrix = torch.vstack((previous_crops_selected_matrix[2:],current_crop_selection_vector))
        else:
            current_crop_selection_vector = torch.zeros((1,self.num_crops)).to(self.device)
            current_crop_selection_vector[0,action] = 1
            previous_crops_selected_matrix = torch.vstack((previous_crops_selected_matrix[1:],current_crop_selection_vector))
        return previous_crops_selected_matrix.flatten()
    

    def train(
            self,
            replay_buffer, 
            num_steps, 
            batch_size,
            ):
        self.model_loss_optimizer_pool, mse_losses, kl_losses = train_all_models(
            model_loss_optimizer_pool = self.model_loss_optimizer_pool, 
            num_steps = num_steps, 
            replay_buffer = replay_buffer,
            batch_size = batch_size,
            device = self.device,
            num_crops = self.num_crops)
        
    def train_single(
            self,
            model_key,
            model_idx,
            replay_buffer,
            num_steps, 
            batch_size,
            ):
        self.model_loss_optimizer_pool, mse_losses, kl_losses = train_single_model(
            model_loss_optimizer_pool = self.model_loss_optimizer_pool,
            model_key = model_key,
            model_idx = model_idx,
            num_steps = num_steps, 
            replay_buffer = replay_buffer,
            batch_size = batch_size,
            device = self.device,
            num_crops = self.num_crops)
        
    # def predict(
    #         self, 
    #         state,
    #         action,
    #         model_idxs,
    #         device
    #         ):
    #     input = format_state_action_for_prediction(
    #          state = state, 
    #          action = action,
    #          device=device, 
    #          num_actions=self.num_crops)
    #     assert input.shape[1] == self.input_size
    #     with torch.no_grad():
    #         for idx_model_key, model_key in enumerate(self.model_loss_optimizer_pool.keys()):
    #             model_idx = model_idxs[idx_model_key]
    #             input_filtered = input[:,self.model_loss_optimizer_pool[model_key]["input_idxs"]]
    #             output = self.model_loss_optimizer_pool[model_key]["models"][model_idx](input_filtered)
    #             if model_key == "reward":
    #                 reward = output
    #             elif model_key == "stochastic_multi":
    #                 stochastic_multi = output
    #             else:
    #                 raise ValueError(f"Unknown model_key: {model_key}")
    #     next_state = self.get_next_state_from_prediction(
    #         state = state, 
    #         action = action, 
    #         predicted_stochastic_multi = stochastic_multi)
    #     assert next_state.shape[1] == self.state_size
    #     return next_state, reward, None
    
    def predict_batch(
            self, 
            states,
            actions,
            model_idxs_all_rollouts,
            device
            ):
        inputs = format_state_action_for_prediction(
             state = states,
             action = actions,
             device=device, 
             num_actions=self.num_crops
             )
        assert inputs.shape[1] == self.input_size
        with torch.no_grad():
            rewards = torch.tensor([]).to(device)
            next_states = torch.tensor([]).to(device)
            for i_rollout_episode, model_idxs in enumerate(model_idxs_all_rollouts):
                for idx_model_key, model_key in enumerate(self.model_loss_optimizer_pool.keys()):
                    model_idx = model_idxs[idx_model_key]
                    input_filtered = inputs[i_rollout_episode,self.model_loss_optimizer_pool[model_key]["input_idxs"]]
                    model = self.model_loss_optimizer_pool[model_key]["models"][model_idx]
                    model.eval()
                    output = model(input_filtered)
                    if model_key == "reward":
                        reward = output
                    elif model_key == "stochastic_multi":
                        stochastic_multi = output
                    else:
                        raise ValueError(f"Unknown model_key: {model_key}")
                next_state = self.get_next_state_from_prediction(
                    state = states[i_rollout_episode], 
                    action = actions[i_rollout_episode], 
                    predicted_stochastic_multi = stochastic_multi)
                rewards = torch.cat((rewards, reward), dim = 0)
                next_states = torch.cat((next_states, next_state), dim = 0)
        assert next_states.shape[1] == self.state_size
        assert next_states.shape[0] == inputs.shape[0]
        return next_states, rewards
    
    def eval(
            self,
            test_inputs,
            test_outputs
            ):
        mean_mse_losses = {}
        mean_kl_losses = {}
        with torch.no_grad():
            for idx_model_key, model_key in enumerate(self.model_loss_optimizer_pool.keys()):
                mse_loss_fn = self.model_loss_optimizer_pool[model_key]["mse_loss"]
                kl_loss_fn = self.model_loss_optimizer_pool[model_key]["kl_loss"]
                test_inputs_filtered = test_inputs[:,self.model_loss_optimizer_pool[model_key]["input_idxs"]]
                test_outputs_filtered = test_outputs[:,self.model_loss_optimizer_pool[model_key]["target_idxs"]]
                mean_mse_losses[model_key] = []
                mean_kl_losses[model_key] = []
                
                for model_idx in range(len(self.model_loss_optimizer_pool[model_key]["models"])):
                    model = self.model_loss_optimizer_pool[model_key]["models"][model_idx]
                    model.eval()
                    pres = []
                    mse_losses_for_model_idx = []
                    kl_losses_for_model_idx = []
                    for i in range(10):
                        pre = model(test_inputs_filtered)
                        pres.append(pre)
                        mse_loss = mse_loss_fn(pre, test_outputs_filtered).item()
                        kl_loss = kl_loss_fn(self.model_loss_optimizer_pool[model_key]["models"][model_idx]).item()
                        mse_losses_for_model_idx.append(mse_loss)
                        kl_losses_for_model_idx.append(kl_loss)
                    mean_mse_loss_for_model_idx = np.mean(mse_losses_for_model_idx)
                    mean_kl_loss_for_model_idx = np.mean(kl_losses_for_model_idx)
                    mean_mse_losses[model_key].append(mean_mse_loss_for_model_idx)
                    mean_kl_losses[model_key].append(mean_kl_loss_for_model_idx)
        return mean_mse_losses, mean_kl_losses
    
    def eval_single_model(
            self,
            model_key,
            model_idx,
            test_inputs,
            test_outputs
            ):
        mean_mse_losses = {}
        mean_kl_losses = {}
        with torch.no_grad():
            mse_loss_fn = self.model_loss_optimizer_pool[model_key]["mse_loss"]
            kl_loss_fn = self.model_loss_optimizer_pool[model_key]["kl_loss"]
            test_inputs_filtered = test_inputs[:,self.model_loss_optimizer_pool[model_key]["input_idxs"]]
            test_outputs_filtered = test_outputs[:,self.model_loss_optimizer_pool[model_key]["target_idxs"]]
            mean_mse_losses[model_key] = []
            mean_kl_losses[model_key] = []
            pres = []
            mse_losses_for_model_idx = []
            kl_losses_for_model_idx = []
            model = self.model_loss_optimizer_pool[model_key]["models"][model_idx]
            model.eval()
            for i in range(10):
                pre = model(test_inputs_filtered)
                pres.append(pre)
                mse_loss = mse_loss_fn(pre, test_outputs_filtered).item()
                kl_loss = kl_loss_fn(self.model_loss_optimizer_pool[model_key]["models"][model_idx]).item()
                mse_losses_for_model_idx.append(mse_loss)
                kl_losses_for_model_idx.append(kl_loss)
            mean_mse_loss_for_model_idx = np.mean(mse_losses_for_model_idx)
            mean_kl_loss_for_model_idx = np.mean(kl_losses_for_model_idx)
            mean_mse_losses[model_key].append(mean_mse_loss_for_model_idx)
            mean_kl_losses[model_key].append(mean_kl_loss_for_model_idx)
        return mean_mse_losses, mean_kl_losses

                
    

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
			import matplotlib.pyplot as plt
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

        