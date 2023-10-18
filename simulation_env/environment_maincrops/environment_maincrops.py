# SET LENGTH HERE
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import random
from .data.mappings import crop_mapping_german, crop_mapping_german_rev, crop_mapping_eng, date_mapping, date_mapping_rev
from .data.cropbreaks import cropbreaks, mf_groups, ap_groups
from .data.kolbe import kolbe_matrix
import json
import os
import pprint

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

class CropRotationEnv(Env):
    def __init__(self, seed = 42, seq_len = 8, NInit = 50.0, region = None, deterministic = True):
        if region:
           self.region = region
        else:
           self.region ="Bundesgebiet"
        
        self.seed = seed
        random.seed(self.seed)
        # German names of crops used in the crop rotation
        self.crop_mapping_german = crop_mapping_german
        self.crop_mapping_german_rev = crop_mapping_german_rev
        
        # English names of crops used in the crop rotation
        self.crop_mapping_eng = crop_mapping_eng


        # Mapping dates to integer indices
        self.date_mapping = date_mapping
        self.date_mapping_rev = date_mapping_rev
        
        # Number of crops
        self.num_crops = len(self.crop_mapping_german)

        # Successor crop suitability matrix, -2: non-suitable combination, -1: rather unsuitable combination, 1: good combination, 2: very good combination
        self.suitability_matrix = kolbe_matrix
        self.suitability_dict = {
          -2: 0.8,
          -1: 0.9,
          1: 1.1,
          2: 1.2
        }

        # Source: LFL
        # Source for minimum humus: Landwirt_Nr._13-2015_-_Wie_viel_Humus_braucht_der_Ackerboden
        self.ground_humus_dict = {
            "initial_humus": {
                -1.0: 2.07,
                0.0: 2.76,
                1.0: 3.45
            },
            "minimum_humus": {
                -1.0: 2.0,
                0.0: 2.5,
                1.0: 3.0
            }
        }
        self.corg_to_humus_percent_factor = 1.72 # LFL Bayern
        self.humus_equivalent_to_percent_factor = 0.0005672 # https://llg.sachsen-anhalt.de/fileadmin/Bibliothek/Politik_und_Verwaltung/MLU/LLFG/Dokumente/04_themen/agraroekologie/2017_humusversorgung-_web-st_.pdf
        self.organic_fertilizer_dict = {
            "nitrogen_per_tonne": 3.3,
            "humus_equivalent_factor":8.0
        }
        self.cropRotationSequenceLengthInit = seq_len
        self.cropRotationSequenceLength = self.cropRotationSequenceLengthInit
        self.NInit = NInit
        self.N = self.NInit
        self.P = 0.0
        self.K = 0.0
        self.Week = 0
        # Select ground type: -1 = light, 0 = medium, 1 = heavy
        self.GroundType = random.choice([-1.0,0.0,1.0])

        # Humus initialization
        self.HumusInit = self.ground_humus_dict["initial_humus"][self.GroundType]
        self.Humus = self.HumusInit
        # Probability to have a wet year
        self.DryWetProb = random.uniform(0.2,0.8)
        # Actual index, 1 = wet year, 0 = dry year
        self.DryWet = self.get_drywet(self.DryWetProb)

       
        # Previous crops initialization
        self.previous_crop = None
        self.previous_crops_selected = np.array([None,None,None,None,None])
        self.previous_crops_selected_matrix = np.zeros((5, self.num_crops))

        # Cultivation breaks
        self.cropbreaks = cropbreaks
        self.mf_groups = mf_groups
        self.ap_groups = ap_groups

        # Crop properties
        self.maincrop_yields = maincrop_yields
        self.maincrop_properties = maincrop_properties
        
        # GBM & average stats for prices and costs
        self.prices_avg = np.array([crop["Verkaufspreis"]["avg"] for crop in self.maincrop_yields.values()])
        self.prices_std = np.array([crop["Verkaufspreis"]["std"] for crop in self.maincrop_yields.values()])
        self.prices = random.normal(self.prices_avg,self.prices_std)
        self.prices_gbm_avg = np.array([crop["Verkaufspreis"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.prices_gbm_std = np.array([crop["Verkaufspreis"]["std_gbm"] for crop in self.maincrop_yields.values()])

        self.sowing_costs_avg = np.array([crop["Kosten"]["Saatgut"]["avg"] for crop in self.maincrop_yields.values()])
        self.sowing_costs_std = np.array([crop["Kosten"]["Saatgut"]["std"] for crop in self.maincrop_yields.values()])
        self.sowing_costs = random.normal(self.sowing_costs_avg,self.sowing_costs_std)
        self.sowing_costs_gbm_avg = np.array([crop["Kosten"]["Saatgut"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.sowing_costs_gbm_std = np.array([crop["Kosten"]["Saatgut"]["std_gbm"] for crop in self.maincrop_yields.values()])

        self.other_costs_avg = np.array([crop["Kosten"]["Sonstiges"]["avg"] for crop in self.maincrop_yields.values()])
        self.other_costs_std = np.array([crop["Kosten"]["Sonstiges"]["std"] for crop in self.maincrop_yields.values()])
        self.other_costs = random.normal(self.other_costs_avg,self.other_costs_std)
        self.other_costs_gbm_avg = np.array([crop["Kosten"]["Sonstiges"]["avg_gbm"] for crop in self.maincrop_yields.values()])
        self.other_costs_gbm_std = np.array([crop["Kosten"]["Sonstiges"]["std_gbm"] for crop in self.maincrop_yields.values()])
        
        self.N_costs_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["avg"]
        self.N_costs_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["std"]
        self.N_costs = random.normal(self.N_costs_avg,self.N_costs_std)
        self.N_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["avg_gbm"]
        self.N_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["std_gbm"]

        self.P_costs_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["avg"]
        self.P_costs_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["std"]
        self.P_costs = random.normal(self.P_costs_avg,self.P_costs_std)
        self.P_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["avg_gbm"]
        self.P_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["std_gbm"]

        self.K_costs_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["avg"]
        self.K_costs_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["std"]
        self.K_costs = random.normal(self.K_costs_avg,self.K_costs_std)
        self.K_costs_gbm_avg = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["avg_gbm"]
        self.K_costs_gbm_std = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["std_gbm"]       

        # Calculate maximum reward
        average_yields = np.array([self.maincrop_yields[crop_name]["Ertrag"][self.region]["avg"] if self.region in self.maincrop_yields[crop_name]["Ertrag"].keys() else self.maincrop_yields[crop_name]["Ertrag"]["Bundesgebiet"]["avg"] for crop_name in self.maincrop_yields.keys()])
        # TODO: Add costs
        # Calculate actual costs for nitrogen, phosphor and kalium fertilizations
        # maximum amount of nitrogen fertilization per hectar is 170 kg
        average_N_fertilization_needs = np.array([max(self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"] - self.N,0.0) for crop_name in self.maincrop_yields.keys()])
        max_N_fertilization_needs = np.ones_like(average_N_fertilization_needs)*170.0
        average_N_fertilization_needs = np.minimum(average_N_fertilization_needs,max_N_fertilization_needs)
        average_P_fertilization_needs = np.array([max(self.maincrop_yields[crop_name]["Duengebedarf P"]["Value"] - self.P,0.0) for crop_name in self.maincrop_yields.keys()])
        average_K_fertilization_needs = np.array([max(self.maincrop_yields[crop_name]["Duengebedarf K"]["Value"] - self.K,0.0) for crop_name in self.maincrop_yields.keys()])

        average_N_fertilization_costs = self.N_costs * average_N_fertilization_needs
        average_P_fertilization_costs = self.P_costs * average_P_fertilization_needs
        average_K_fertilization_costs = self.K_costs * average_K_fertilization_needs

        # Calculate profits
        average_profits = average_yields * self.prices_avg - self.sowing_costs_avg - self.other_costs_avg - average_N_fertilization_costs - average_P_fertilization_costs - average_K_fertilization_costs
        average_costs = self.sowing_costs_avg + self.other_costs_avg + average_N_fertilization_costs + average_P_fertilization_costs + average_K_fertilization_costs
        max_reward_idx = average_profits.argmax()
        min_reward_idx = average_costs.argmax()
        self.max_reward = average_profits[max_reward_idx]
        self.min_reward = -(average_costs[min_reward_idx])


        # Action and observation space are defined by crops in cropNames
        # State size = 1 (0: Nitrogen-Level) + 1 (1: Phosphor-Level) + 1 (2: Kalium-Level) + 1 (3: Week index of year) + 1 (4: Ground type) + 1 (5: Dry/Wet year) + 1 (6: Humus %) + 3 (7-9: fertilizer costs) \n
        # 3*num_crops (prices[10:10+num_crops] + sowing[10+num_crops:10+num_crops*2] + other costs[10+num_crops*2:10+num_crops*3]) + 5*num_crops (previous crops selected) = 192
        self.state_size = 10 + 3*self.num_crops + 5*self.num_crops
        self.low = np.zeros(self.state_size)
        self.high = np.ones(self.state_size)
        self.maximum_prices = np.array([crop["Verkaufspreis"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_sowing_costs = np.array([crop["Kosten"]["Saatgut"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_other_costs = np.array([crop["Kosten"]["Sonstiges"]["max"] for crop in self.maincrop_yields.values()])*2
        self.maximum_N_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger N"]["max"]*2
        self.maximum_P_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger P"]["max"]*2
        self.maximum_K_costs = self.maincrop_yields["WINTERWEIZEN"]["Kosten"]["Duenger K"]["max"]*2
        # Nitrogen, Phosphor, Kalium
        self.high[0] = float(400) # maximum attainable nitrogen levels + some buffer
        self.high[1] = float(400) # maximum attainable phosphor levels + some buffer
        self.high[2] = float(400) # maximum attainable phosphor levels + some buffer
        # Week index
        self.high[3] =  float(35)
        # Ground/type (-1, 0, 1)
        self.low[4] =  float(-1)
        self.high[4] =  float(1)
        # Wet/dry = one-hot-encoded and therefore 0,1
        # Humus %
        self.high[6] =  float(15)
        # Fertilizer costs (at least double the maximum cost from actual data)
        self.high[7] = self.maximum_N_costs
        self.high[8] = self.maximum_P_costs
        self.high[9] = self.maximum_K_costs
        # Prices
        self.high[10:(10+self.num_crops)] = self.maximum_prices
        self.high[(10+self.num_crops):(10+self.num_crops*2)] = self.maximum_sowing_costs
        self.high[(10+self.num_crops*2):(10+self.num_crops*3)] = self.maximum_other_costs
        # The previous crop selection is one-hot-encoded and therefore stays between 0 and 1


        # Define observation and action space
        self.observation_space = Box(low=self.low, high=self.high, shape = (len(self.high),), dtype=np.float32)
        self.action_space = Discrete(self.num_crops) # Discrete actions: Select a crop to grow
        
        # Initial state
        self.state = np.concatenate((np.array([self.N,self.P, self.K, self.Week, self.GroundType, self.DryWet, self.Humus,self.N_costs,self.P_costs,self.K_costs]), self.prices, self.sowing_costs, self.other_costs,self.previous_crops_selected_matrix.flatten())) 
        self.state_normalized = np.concatenate((np.array([
                    (self.N-self.low[0])/(self.high[0]-self.low[0]),
                    (self.P-self.low[1])/(self.high[1]-self.low[1]),
                    (self.K-self.low[2])/(self.high[2]-self.low[2]), 
                    (self.Week-self.low[3])/(self.high[3]-self.low[3]),
                    (self.GroundType-self.low[4])/(self.high[4]-self.low[4]),
                    (self.DryWet-self.low[5])/(self.high[5]-self.low[5]),
                    (self.Humus-self.low[6])/(self.high[6]-self.low[6]),
                    (self.N_costs-self.low[7])/(self.high[7]-self.low[7]),
                    (self.P_costs-self.low[8])/(self.high[8]-self.low[8]),
                    (self.K_costs-self.low[9])/(self.high[9]-self.low[9])]),
                (self.prices-self.low[10:10+self.num_crops])/(self.high[10:10+self.num_crops]-self.low[10:10+self.num_crops]),
                (self.sowing_costs-self.low[10+self.num_crops:10+2*self.num_crops])/(self.high[10+self.num_crops:10+2*self.num_crops]-self.low[10+self.num_crops:10+2*self.num_crops]),
                (self.other_costs-self.low[10+2*self.num_crops:10+3*self.num_crops])/(self.high[10+2*self.num_crops:10+3*self.num_crops]-self.low[10+2*self.num_crops:10+3*self.num_crops]),
                (self.previous_crops_selected_matrix.flatten())))
        # Initial crop of the crop sequence is selected randomly; initial state is stored for later usage
        # self.state_index = random.randint(0,len(self.cropNamesEN)-1)
        # self.initial_index = self.state_index
        # self.state = convert_index_to_state(len(self.cropNamesEN),self.state_index)
        # self.initial = self.state
        # # Set the last cultivation step for the current crop to 1
        # self.cropLastCultivationList[self.state_index] = 1
        # Soil is initialized with a nitrogen level of NMin; current yield and reward are set to 0; crop rotation sequence length set to 10
        self.current_yield = 0
        self.reward = 0.0
        self.info = {
            "Previous crops": self.previous_crops_selected,
            "Last profit" : self.reward,
            "N-Level" : self.N,
            "P-Level" : self.P,
            "K-Level" : self.K,
            "Week": (self.Week,date_mapping[self.Week]),
            "Ground Type": self.GroundType,
            "Dry/Wet": self.DryWet,
            "Humus": self.Humus,
        }

    def calculate_next_yield(self, avg, std):
        return random.normal(avg,std)

    def calculate_next_price(self, last_value, avg_gbm, std_gbm):
        return last_value * np.exp((avg_gbm - (std_gbm ** 2) / 2) + std_gbm * random.normal())

    def get_drywet(self, threshold):
        rn = random.uniform(0,1)
        if rn <= threshold:
          return 1.0
        else:
          return 0.0
    def get_single_AP_penalty(self, last_crops_orig,action,AP):
        penalty = 1.0
        last_crops = np.append(last_crops_orig,[action])
        i = 0
        while len(last_crops)-i-1 >= 0:
            window_start = max(0,len(last_crops)-AP-i-1)
            window_end = len(last_crops)-i-1
            if last_crops[window_end] != action:
                i+=1
                continue
            break_indices = np.where(last_crops[window_start:window_end]==action)[0]
            if len(break_indices) != 0:
                last_break_index = break_indices[-1]
                penalty *= 1-0.1*AP*(last_break_index+1)/AP
                i+=1
            else:
                i+=1
                break
        return np.round(penalty,2)
    def get_multi_AP_penalty(self, last_crops_orig,action,AP, relevant_crops):
        penalty = 1.0
        last_crops = np.append(last_crops_orig,[action])
        i = 0
        while len(last_crops)-i-1 >= 0:
            window_start = max(0,len(last_crops)-AP-i-1)
            window_end = len(last_crops)-i-1
            if last_crops[window_end] not in relevant_crops:
                i+=1
                continue
            break_indices = np.where(np.isin(last_crops[window_start:window_end],relevant_crops))[0]
            if len(break_indices) != 0:
                last_break_index = break_indices[-1]
                penalty *= 1-0.1*AP*(last_break_index+1)/AP
                i+=1
            else:
                i+=1
                break
        return np.round(penalty,2)
    
    def step(self, action):
        # Store inital crop for later usage
        # self.initial = self.state
        # self.initial_index = self.state_index
        # Get suitability of crop combination and determine yield; add 10% if it is a suitable combination and 20% if it is a very suitable combination
        # subtract 10% if rather unsuitable, subtract 20% if very unsuitable
        crop_name = self.crop_mapping_german[action]

        if self.region in self.maincrop_yields[crop_name]["Ertrag"]:
            self.current_yield = self.calculate_next_yield(self.maincrop_yields[crop_name]["Ertrag"][self.region]["avg"], self.maincrop_yields[crop_name]["Ertrag"][self.region]["std"])
        else:
            self.current_yield = self.calculate_next_yield(self.maincrop_yields[crop_name]["Ertrag"]["Bundesgebiet"]["avg"], self.maincrop_yields[crop_name]["Ertrag"]["Bundesgebiet"]["std"])

        if self.Humus < self.ground_humus_dict["minimum_humus"][self.GroundType]:
            humus_yield_factor = max(0.77,1.0+(self.Humus-self.ground_humus_dict["minimum_humus"][self.GroundType])*0.38)
        else:
            humus_yield_factor = 1.0
        suitability_break_flag = False

        self.current_yield = humus_yield_factor*self.current_yield

        # Calculate crop combination factor (previous crop -> current crop)
        crop_combination_factor = 1.0
        if self.previous_crop:
            cropCombinationSuitability = self.suitability_matrix["PRECROP"][self.previous_crop][action]
            crop_combination_factor = self.suitability_dict[cropCombinationSuitability]
            if crop_combination_factor < 1.0:
                suitability_break_flag = True


        crop_break_factors = []
        crop_break_stats = self.cropbreaks[action]
        crop_break_count = 0

        # Check if the crop's individual break was violated
        # Function is penalizing consecutive rule violations higher than single rule violations 
        # and violations with low distances but high necessary breaks higher too (up to very small values < 0.2)
        # Example for action = 1 and AP = 3:
        # [None None None None None] 1.0
        # [1 1 1 1 1] 0.247
        # [None 3 3 1 1] 0.49
        # [None None 1 3 1] 0.56
        # [None 1 3 1 1] 0.392
        # [None None 3 1 3] 0.8
        # [3 1 1 3 3] 0.72
        # [1 1 1 3 1] 0.403
        crop_ap = int(crop_break_stats["AP"])
        crop_break_factors.append(self.get_single_AP_penalty(self.previous_crops_selected,action,crop_ap))
        if crop_break_factors[-1] < 1.0:
            crop_break_count+=1
        
        # Check if crop is breaking MF rules
        # following the scheme:
        # (2, 1, 2) 0.5
        # (2, 1, 3) 0.66
        # (3, 1, 3) 0.33
        # (2, 1, 4) 0.75
        # (3, 1, 4) 0.5
        # (4, 1, 4) 0.25
        # (3, 2, 4) 0.75
        # (4, 2, 4) 0.5
        # (4, 3, 4) 0.75
        crop_mf_groups = crop_break_stats["MF Groups"]
        for mf_group in crop_mf_groups:
            count = 1
            mf_group_indices = self.mf_groups["group_indices"][mf_group]
            mf_group_max_count, mf_group_window_length = self.mf_groups["max_frequencies"][mf_group]
            for previous_crop in self.previous_crops_selected[-mf_group_window_length:]:
                if previous_crop in mf_group_indices:
                    count+=1
            if count>mf_group_max_count:
                crop_break_count+=1
                crop_break_mf_factor = np.round(1-(count/mf_group_window_length - mf_group_max_count/mf_group_window_length),2)                
                crop_break_factors.append(crop_break_mf_factor)

        # Check if the crop's AP_group break rules were violated
        # Function is penalizing consecutive rule violations higher than single rule violations 
        # and violations with low distances but high necessary breaks higher too (up to very small values < 0.2)
        # Example for action = 1, AP = 3 and relevant crops = [1,2,4]:
        # [None None None None None] 1.0
        # [1 1 1 1 1] 0.247
        # [1 2 2 1 1] 0.247
        # [None 3 3 2 1] 0.49
        # [None 3 3 4 1] 0.49
        # [None None 2 3 1] 0.56
        # [None 4 3 2 1] 0.392
        # [None None 3 4 3] 0.8
        # [3 2 1 3 3] 0.72
        # [4 4 2 3 1] 0.403
        crop_ap_groups = crop_break_stats["AP Groups"]
        for ap_group in crop_ap_groups:
            ap_group_indices = self.ap_groups["group_indices"][ap_group]
            crop_ap_group_min_break = self.ap_groups["min_breaks"][ap_group]
            crop_break_ap_factor = self.get_multi_AP_penalty(self.previous_crops_selected,action,crop_ap_group_min_break, ap_group_indices)
            if crop_break_ap_factor < 1.0:
                crop_break_count+=1
            crop_break_factors.append(crop_break_ap_factor)
        

        # aggregate the crop break factors by multiplication
        # TODO: change back
        crop_break_factor = np.multiply.reduce(crop_break_factors)
        # crop_break_factor = min(crop_break_factors)
        
        timing_violation_flag = False
        # calculating the timing factor
        # subtract 0.2 for each week the crop is sown too late (including one week of buffer needed for ground operation & sowing)
        if self.maincrop_properties[crop_name]["summercrop"] == 1 or self.Week < self.date_mapping_rev[self.maincrop_properties[crop_name]["latest sowing"]]-1:
            timing_factor = 1.0
        else:
            timing_violation_flag = True
            timing_factor = max(0.,1+(self.date_mapping_rev[self.maincrop_properties[crop_name]["latest sowing"]] - 1 - self.Week)*0.2)

        ground_type_violation_flag = False
        # calculating the ground factor
        if self.GroundType in self.maincrop_properties[crop_name]["ground type"]:
            ground_factor = 1.0
        else:
            ground_type_violation_flag = True
            ground_factor = 0.9
        
        drywet_violation_flag = False
        # Finding out if the previous season was dry or wet
        # self.DryWet = self.get_drywet(self.DryWetProb)
        # calculating the dry/wet factor
        if self.DryWet == 1.0:
            drywet_factor = 1.1
        else:
            if self.maincrop_properties[crop_name]["drought resistance"] == 1.0:
                drywet_factor = 1.0
            elif self.maincrop_properties[crop_name]["drought resistance"] == 0.0:
                drywet_violation_flag = True
                drywet_factor = 0.9
            elif self.maincrop_properties[crop_name]["drought resistance"] == -1.0:
                drywet_violation_flag = True
                drywet_factor = 0.8
        
        # maximum amount of nitrogen fertilization per hectar is 170 kg
        self.N_fertilization_need = min(max(self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"]*self.current_yield - self.N,0.),170.0)
        self.P_fertilization_need = max(self.maincrop_yields[crop_name]["Duengebedarf P"]["Value"]*self.current_yield - self.P,0.)
        self.K_fertilization_need = max(self.maincrop_yields[crop_name]["Duengebedarf K"]["Value"]*self.current_yield - self.K,0.)

        self.N = self.N + self.N_fertilization_need
        self.P = self.P + self.P_fertilization_need
        self.K = self.K + self.K_fertilization_need

        # calculate humus yield effect
        # https://www.lfl.bayern.de/iab/boden/031146/#:~:text=Ergebnisse%20von%20Dauerfeldversuchen%20zeigen%2C%20dass,6%20und%207%20exemplarisch%20dargestellt.

        # TODO remove setting factors to 1
        # crop_break_factor = 1.0

        # calculate yields via GBM
        # yield is penalized by different factors representing rule violations
        # actual yield can not be higher than yield possible from available nitrogen taken up by plants
        total_reduction_factor = humus_yield_factor * crop_combination_factor * crop_break_factor * timing_factor * ground_factor * drywet_factor
        unknown_reduction_factor = crop_combination_factor * crop_break_factor * timing_factor * ground_factor * drywet_factor
        if self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"] > 0.0:
            actual_yield = min(self.current_yield * unknown_reduction_factor, self.N/self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"])
        else:
            actual_yield = self.current_yield * unknown_reduction_factor
        
        # TODO remove this
        actual_yield = self.current_yield * unknown_reduction_factor
        if total_reduction_factor < 0.4:
            actual_yield = 0.0


        # Calculate actual costs for nitrogen, phosphor and kalium fertilizations
        total_N_fertilization_cost = self.N_costs * self.N_fertilization_need
        total_P_fertilization_cost = self.P_costs * self.P_fertilization_need
        total_K_fertilization_cost = self.K_costs * self.K_fertilization_need

        # Calculate profit
        price = self.prices[action]
        sowing_costs = self.sowing_costs[action]
        other_costs = self.other_costs[action]
        profit = actual_yield * self.prices[action] - self.sowing_costs[action] - self.other_costs[action] - total_N_fertilization_cost - total_P_fertilization_cost - total_K_fertilization_cost
        # Update nitrogen, phosphor and kalium levels
        humus_postdelivery = 20.0 if self.Humus >= 4.0 else 0.0
        if self.previous_crop:
            precrop_postdelivery = self.maincrop_properties[self.crop_mapping_german[self.previous_crop]]["postdelivery"]
        else:
            precrop_postdelivery = 0.0


        
        # Update of state
        if actual_yield <=self.current_yield:
            self.N = self.N - actual_yield*self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"] + humus_postdelivery + precrop_postdelivery + self.maincrop_yields[crop_name]["N-Fixierung"]["Value"]*actual_yield
            self.P = self.P - actual_yield*self.maincrop_yields[crop_name]["Duengebedarf P"]["Value"]
            self.K = self.K - actual_yield*self.maincrop_yields[crop_name]["Duengebedarf K"]["Value"]
        else:
            self.N = self.N - self.current_yield*self.maincrop_yields[crop_name]["Duengebedarf N"]["Value"] + humus_postdelivery + precrop_postdelivery + self.maincrop_yields[crop_name]["N-Fixierung"]["Value"]*self.current_yield
            self.P = self.P - self.current_yield*self.maincrop_yields[crop_name]["Duengebedarf P"]["Value"]
            self.K = self.K - self.current_yield*self.maincrop_yields[crop_name]["Duengebedarf K"]["Value"]
        # Humus buildup/loss from selected crop
        self.Humus += self.maincrop_properties[crop_name]["humus equivalent"]*total_reduction_factor*self.humus_equivalent_to_percent_factor*self.corg_to_humus_percent_factor
        # Humus buildup from organic fertilizer ("cattle manure 6% dry mass")
        self.Humus += self.N_fertilization_need/self.organic_fertilizer_dict["nitrogen_per_tonne"]*self.organic_fertilizer_dict["humus_equivalent_factor"]*self.humus_equivalent_to_percent_factor*self.corg_to_humus_percent_factor
        if action in [0,1]:
            self.cropRotationSequenceLength -=2
        else:
            self.cropRotationSequenceLength -=1
        self.Week = self.date_mapping_rev[self.maincrop_properties[crop_name]["earliest harvest"]]
        

        # Set to done if crop rotation sequence is finished
        if self.cropRotationSequenceLength <= 0: 
            done = True
        else:
            done = False
        
        # Update previous selected crops for state
        current_crop_selection_vector = np.zeros((1,self.previous_crops_selected_matrix.shape[1]))
        current_crop_selection_vector[0,action] = 1
        self.previous_crops_selected_matrix = np.vstack((self.previous_crops_selected_matrix[1:],current_crop_selection_vector))
        self.previous_crop = action
        self.previous_crops_selected = np.append(self.previous_crops_selected[1:], action)


        # Pushed this to the back to be more deterministic
        # TODO Add this again to have switching DryWet Probabilities.
        # self.DryWet = self.get_drywet(self.DryWetProb)
        
        # Calculate next costs and prices
        self.prices = self.calculate_next_price(self.prices, self.prices_gbm_avg, self.prices_gbm_std)
        self.sowing_costs = self.calculate_next_price(self.sowing_costs, self.sowing_costs_gbm_avg, self.sowing_costs_gbm_std)
        self.other_costs = self.calculate_next_price(self.other_costs, self.other_costs_gbm_avg, self.other_costs_gbm_std)
        self.N_costs = self.calculate_next_price(self.N_costs, self.N_costs_gbm_avg, self.N_costs_gbm_std)
        self.P_costs = self.calculate_next_price(self.P_costs, self.P_costs_gbm_avg, self.P_costs_gbm_std)
        self.K_costs = self.calculate_next_price(self.K_costs, self.K_costs_gbm_avg, self.K_costs_gbm_std)       


        # Limit prices and costs to be <= maximum prices and costs
        self.prices = np.minimum(self.prices, self.maximum_prices)
        self.sowing_costs = np.minimum(self.sowing_costs, self.maximum_sowing_costs)
        self.other_costs = np.minimum(self.other_costs, self.maximum_other_costs)
        self.N_costs = np.minimum(self.N_costs, self.maximum_N_costs)
        self.P_costs = np.minimum(self.P_costs, self.maximum_P_costs)
        self.K_costs = np.minimum(self.K_costs, self.maximum_K_costs)
        
        
        self.state = np.concatenate((np.array([self.N,self.P, self.K, self.Week, self.GroundType, self.DryWet, self.Humus,self.N_costs,self.P_costs,self.K_costs]), self.prices, self.sowing_costs, self.other_costs,self.previous_crops_selected_matrix.flatten())) 
        self.state_normalized = np.concatenate((np.array([
                    (self.N-self.low[0])/(self.high[0]-self.low[0]),
                    (self.P-self.low[1])/(self.high[1]-self.low[1]),
                    (self.K-self.low[2])/(self.high[2]-self.low[2]), 
                    (self.Week-self.low[3])/(self.high[3]-self.low[3]),
                    (self.GroundType-self.low[4])/(self.high[4]-self.low[4]),
                    (self.DryWet-self.low[5])/(self.high[5]-self.low[5]),
                    (self.Humus-self.low[6])/(self.high[6]-self.low[6]),
                    (self.N_costs-self.low[7])/(self.high[7]-self.low[7]),
                    (self.P_costs-self.low[8])/(self.high[8]-self.low[8]),
                    (self.K_costs-self.low[9])/(self.high[9]-self.low[9])]),
                (self.prices-self.low[10:10+self.num_crops])/(self.high[10:10+self.num_crops]-self.low[10:10+self.num_crops]),
                (self.sowing_costs-self.low[10+self.num_crops:10+2*self.num_crops])/(self.high[10+self.num_crops:10+2*self.num_crops]-self.low[10+self.num_crops:10+2*self.num_crops]),
                (self.other_costs-self.low[10+2*self.num_crops:10+3*self.num_crops])/(self.high[10+2*self.num_crops:10+3*self.num_crops]-self.low[10+2*self.num_crops:10+3*self.num_crops]),
                (self.previous_crops_selected_matrix.flatten())))
        
        # pp.pprint(self.state_normalized)
        self.filter_information = np.concatenate((np.array([self.Week, self.GroundType, self.DryWet]),np.array(self.previous_crops_selected)))

        # Apply reward and step information
        self.reward = profit/(self.max_reward-self.min_reward)
        # print("Crop reduction factors: ",humus_yield_factor, crop_combination_factor, crop_break_factor, timing_factor, ground_factor, drywet_factor)
        # print("% of actual yield",actual_yield/self.current_yield, "Reward: ",self.reward)
        self.info = {
            "Previous crop": self.crop_mapping_eng[action],
            "Previous crops": self.previous_crops_selected,
            "Last profit" : self.reward,
            "N-Level" : self.N,
            "P-Level" : self.P,
            "K-Level" : self.K,
            "Week": (self.Week,date_mapping[self.Week]),
            "Ground Type": self.GroundType,
            "Dry/Wet": self.DryWet,
            "Humus": self.Humus,

            "Humus Yield Factor": humus_yield_factor,
            "Crop Combination Factor": crop_combination_factor,
            "Crop Break Factor": crop_break_factor,
            "Crop Break Count": crop_break_count,
            "Timing Factor": timing_factor,
            "Ground Factor": ground_factor,
            "Dry/Wet Factor": drywet_factor,
            "Total Reduction Factor": total_reduction_factor,
            "Num broken rules": crop_break_count + timing_violation_flag,
        }
        
        # Return step information
        return self.state_normalized, self.filter_information, self.reward, done, self.info
    
    # Render human readable information
    def render(self, mode='human'):
        print(self.info)

    # Reset environment
    def reset(self):

        # Plot characteristics
        self.N = self.NInit
        self.P = 0.0
        self.K = 0.0
        self.Humus = self.HumusInit
        self.Week = 0
        self.cropRotationSequenceLength = self.cropRotationSequenceLengthInit
        self.GroundType = random.choice([-1.0,0.0,1.0])
        self.DryWetProb = random.uniform(0.2,0.8) # maybe change to stay the same
        self.DryWet = self.get_drywet(self.DryWetProb)

        # Reset previous crops
        self.previous_crop = None
        self.previous_crops_selected = np.array([None,None,None,None,None])
        self.previous_crops_selected_matrix = np.zeros((5, self.num_crops))

        # Prices & Costs
        # GBM & average stats for prices and costs
        self.prices = random.normal(self.prices_avg,self.prices_std)
        self.sowing_costs = random.normal(self.sowing_costs_avg,self.sowing_costs_std)
        self.other_costs = random.normal(self.other_costs_avg,self.other_costs_std)
        self.N_costs = random.normal(self.N_costs_avg,self.N_costs_std)
        self.P_costs = random.normal(self.P_costs_avg,self.P_costs_std)
        self.K_costs = random.normal(self.K_costs_avg,self.K_costs_std)

        # Set state and reset yield and reward
        self.state = np.concatenate((np.array([self.N,self.P, self.K, self.Week, self.GroundType, self.DryWet, self.Humus,self.N_costs,self.P_costs,self.K_costs]), self.prices, self.sowing_costs, self.other_costs,self.previous_crops_selected_matrix.flatten())) 
        self.state_normalized = np.concatenate((np.array([
                    (self.N-self.low[0])/(self.high[0]-self.low[0]),
                    (self.P-self.low[1])/(self.high[1]-self.low[1]),
                    (self.K-self.low[2])/(self.high[2]-self.low[2]), 
                    (self.Week-self.low[3])/(self.high[3]-self.low[3]),
                    (self.GroundType-self.low[4])/(self.high[4]-self.low[4]),
                    (self.DryWet-self.low[5])/(self.high[5]-self.low[5]),
                    (self.Humus-self.low[6])/(self.high[6]-self.low[6]),
                    (self.N_costs-self.low[7])/(self.high[7]-self.low[7]),
                    (self.P_costs-self.low[8])/(self.high[8]-self.low[8]),
                    (self.K_costs-self.low[9])/(self.high[9]-self.low[9])]),
                (self.prices-self.low[10:10+self.num_crops])/(self.high[10:10+self.num_crops]-self.low[10:10+self.num_crops]),
                (self.sowing_costs-self.low[10+self.num_crops:10+2*self.num_crops])/(self.high[10+self.num_crops:10+2*self.num_crops]-self.low[10+self.num_crops:10+2*self.num_crops]),
                (self.other_costs-self.low[10+2*self.num_crops:10+3*self.num_crops])/(self.high[10+2*self.num_crops:10+3*self.num_crops]-self.low[10+2*self.num_crops:10+3*self.num_crops]),
                (self.previous_crops_selected_matrix.flatten())))
        self.filter_information = np.concatenate((np.array([self.Week, self.GroundType, self.DryWet]),np.array(self.previous_crops_selected)))
        self.current_yield = 0
        self.reward = 0.0
        self.info = {
            "Previous crops": self.previous_crops_selected,
            "Last profit" : self.reward,
            "N-Level" : self.N,
            "P-Level" : self.P,
            "K-Level" : self.K,
            "Week": (self.Week,date_mapping[self.Week]),
            "Ground Type": self.GroundType,
            "Dry/Wet": self.DryWet,
            "Humus": self.Humus,
        }

        return self.state_normalized, self.filter_information