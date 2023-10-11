# SET LENGTH HERE
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import random
from .data.mappings import crop_mapping_german, crop_mapping_german_rev, crop_mapping_eng, date_mapping, date_mapping_rev
from .data.cropbreaks import cropbreaks, mf_groups, ap_groups
from data.kolbe import kolbe_matrix
cropLastCultivationList

def convert_state_to_index(state):
   return np.argmax(state)

def convert_index_to_state(length, index):
  state = np.zeros(length)
  state[index] = 1.0
  return state

json_file_path = "./environment_maincrops/data/maincrops_updated.json"
# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    # Parse the JSON data into a Python dictionary
    maincrop_yields = json.load(json_file)

json_file_path = "./environment_maincrops/data/maincrop_properties.json"
# Open and read the JSON file
with open(json_file_path, "r") as json_file:
    # Parse the JSON data into a Python dictionary
    maincrop_properties = json.load(json_file)

class CropRotationEnv(Env):
    def __init__(self, seed = 42, seq_len = 8, NStart = 50, deterministic = True):
        random.seed(seed)
        self.cropRotationSequenceLengthStatic = seq_len
        self.N = NStart

        # German names of crops used in the crop rotation
        self.crop_mapping_german = crop_mapping_german
        self.crop_mapping_german_rev = crop_mapping_german_rev
        
        # English names of crops used in the crop rotation
        self.crop_mapping_eng = crop_mapping_eng


        # Mapping dates to integer indices
        self.date_mapping = date_mapping
        self.date_mapping_rev = date_mapping_rev
        
        # Successor crop suitability matrix, -2: non-suitable combination, -1: rather unsuitable combination, 1: good combination, 2: very good combination
        self.suitability_matrix = kolbe_matrix

        # Number of crops
        self.num_crops = len(self.crop_mapping_german)
        
        # Previous crops initialization
        self.last_crop_selected = None
        self.previous_crops_selected = np.zeros(self.cropRotationSequenceLengthStatic, self.num_crops)

        # Cultivation breaks
        self.cropsbreaks = cropbreaks
        self.mf_groups = mf_groups
        self.ap_groups = ap_groups

        # Crop properties
        self.maincrop_yields = maincrop_yields
        self.maincrop_properties = maincrop_properties
        
        # GBM & average stats for prices and costs
        self.prices_avg = np.array([crop["Verkaufspreis"]["avg"] for crop in maincrop_yields.items()])
        self.prices.std = np.array([crop["Verkaufspreis"]["std"] for crop in maincrop_yields.items()])
        self.prices = random.normal(self.prices_avg,self.prices.std)
        self.prices_gbmstats = np.array([(crop["Verkaufspreis"]["gbm_avg"],crop["Verkaufspreis"]["gbm_std"]) for crop in maincrop_yields.items()])

        self.sowing_costs_avg = np.array([crop["Kosten"]["Saatgut"]["avg"] for crop in maincrop_yields.items()])
        self.sowing_costs.std = np.array([crop["Kosten"]["Saatgut"]["std"] for crop in maincrop_yields.items()])
        self.sowing_costs = random.normal(self.sowing_costs_avg,self.sowing_costs.std)
        self.sowing_costs_gbmstats = np.array([(crop["Kosten"]["Saatgut"]["gbm_avg"],crop["Kosten"]["Saatgut"]["gbm_std"]) for crop in maincrop_yields.items()])

        self.other_costs_avg = np.array([crop["Kosten"]["Sonstiges"]["avg"] for crop in maincrop_yields.items()])
        self.other_costs.std = np.array([crop["Kosten"]["Sonstiges"]["std"] for crop in maincrop_yields.items()])
        self.other_costs = random.normal(self.other_costs_avg,self.other_costs.std)
        self.other_costs_gbmstats = np.array([(crop["Kosten"]["Sonstiges"]["gbm_avg"],crop["Kosten"]["Sonstiges"]["gbm_std"]) for crop in maincrop_yields.items()])

        self.N_costs_avg = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger N"]["avg"]
        self.N_costs.std = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger N"]["std"]
        self.N_costs = random.normal(self.N_costs_avg,self.N_costs.std)
        self.N_costs_gbmstats = (maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger N"]["gbm_avg"],maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger N"]["gbm_std"])

        self.P_costs_avg = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger P"]["avg"]
        self.P_costs.std = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger P"]["std"]
        self.P_costs = random.normal(self.P_costs_avg,self.P_costs.std)
        self.P_costs_gbmstats = (maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger P"]["gbm_avg"],maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger P"]["gbm_std"])

        self.K_costs_avg = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger K"]["avg"]
        self.K_costs.std = maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger K"]["std"]
        self.K_costs = random.normal(self.K_costs_avg,self.K_costs.std)
        self.K_costs_gbmstats = (maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger K"]["gbm_avg"],maincrop_yields["WINTERWEIZEN"]["Kosten"]["Dünger K"]["gbm_std"])        


        # Action and observation space are defined by crops in cropNames
        # State size = 1 (Nitrogen-Level) + 1 (Week index of year) + 1 (Ground type) + 1 Dry/Wet year \n
        # + 1 Humus % + num_crops (prices) + 2*num_crops (sowing + other costs) + 3 (fertilizer costs) + 
        self.state_size = 1 + 
        self.action_space = Discrete(self.num_crops) # Discrete actions: Select a crop to grow
        self.observation_space = Box(low=np.zeros(len(self.cropNamesEN)), high=np.ones(len(self.cropNamesEN)), shape = (len(self.cropNamesEN),), dtype=np.int16)
        
        # Initial crop of the crop sequence is selected randomly; initial state is stored for later usage
        self.state_index = random.randint(0,len(self.cropNamesEN)-1)
        self.initial_index = self.state_index
        self.state = convert_index_to_state(len(self.cropNamesEN),self.state_index)
        self.initial = self.state
        

        # Soil is initialized with a nitrogen level of NMin; current yield and reward are set to 0; crop rotation sequence length set to 10
        self.currentYield = 0
        self.reward = 0
        
        
        # Set the last cultivation step for the current crop to 1
        self.cropLastCultivationList[self.state_index] = 1

    def calculate_next_yield(self, avg, std):
        return random.normal(avg,std)

    def calculate_next_price(self, last_value, avg_gbm, std_gbm):
        return last_value * np.exp((avg_gbm - (std_gbm ** 2) / 2) + std_gbm * random.normal())

    def step(self, action):
        # Store inital crop for later usage
        self.initial = self.state
        self.initial_index = self.state_index
        # Get suitability of crop combination and determine yield; add 10% if it is a suitable combination and 20% if it is a very suitable combination
        cropCombinationSuitability = self.suitabilityMatrix[self.state_index][action]


        # Yield = 0 if crops are not suitable
        if cropCombinationSuitability == -1:
            self.currentYield = 0
        # Yield = multiplied by 1.1 if crops are suitable=1
        elif cropCombinationSuitability == 1:
            self.currentYield = self.cropYieldList[action]*1.1 #random.uniform(1.0, 1.1) #1.1
        # Yield = multiplied by 1.2 if crops are very suitable=2
        elif cropCombinationSuitability == 2:
            self.currentYield = self.cropYieldList[action]*1.2 #random.uniform(1.1, 1.2) #1.2


        # Add nitrogen consumption/addition of the current crop to the soil nitrogen level
        self.soilNitrogenLevel += self.soilNitrogenList[action]

        # Determine if row culture rule is violated
        # Explanation: A root crop should always be followed by a non-root-crop!
        root_crop_rule_violated = False
        if self.cropRootCropList[self.state_index] == 1 and self.cropRootCropList[action] == 1:
            root_crop_rule_violated = True
        
        # Determine if crop break rule is violated
        crop_break_rule_violated = False
        self.cropRotationSequenceLength -= 1
        if self.cropLastCultivationList[action] > -1 and self.cropRotationSequenceLengthStatic - self.cropRotationSequenceLength < self.cropLastCultivationList[action] + self.cropCultivationBreakList[action]+1:
            crop_break_rule_violated = True
        
        #crop_break_rule_violated = False # nach Test entfernen
        # Increase the crop cultivation counter for the current crop by 1
        # This increase cultivation counter for each legume by 1 and resets LastCultivationList if at least one legume has been planted. 
        if self.cropIsLegumeList[action] == 1:
            for i in range(len(self.cropIsLegumeList)):
                if self.cropIsLegumeList[i] == 1:
                    self.cropCultivationCounterList[i] += 1
                    self.cropLastCultivationList[i] = self.cropRotationSequenceLengthStatic - self.cropRotationSequenceLength
        else:
            self.cropCultivationCounterList[action] += 1

        # Determine if maximum crop occurence rule is violated
        max_crop_occ_rule_violated = False
        # if the maximum number of cultivations per sequence is important and the counter is higher than the maximum number, the rule is violated.
        if self.cropMaxCultivationTimesList[action] > -1 and self.cropCultivationCounterList[action] > self.cropMaxCultivationTimesList[action]:
            max_crop_occ_rule_violated = True

        # Store last cultivation of current crop
        self.cropLastCultivationList[action] = self.cropRotationSequenceLengthStatic - self.cropRotationSequenceLength

        # Add yield to reward if soil and crop combination suitability are ok and no rules have been violated
        if self.soilNitrogenLevel >=0 and cropCombinationSuitability > 0 and crop_break_rule_violated == False and max_crop_occ_rule_violated == False and root_crop_rule_violated == False:
            #if self.soilNitrogenLevel >=0 and cropCombinationSuitability > 0 and max_crop_occ_rule_violated == False and root_crop_rule_violated == False:
            reward = self.currentYield/(max(self.cropYieldList.values())*1.2-self.negativeReward)
        else: 
            reward = self.negativeReward/(max(self.cropYieldList.values())*1.2-self.negativeReward)
        
        # Set to done if crop rotation sequence is finished
        if self.cropRotationSequenceLength <= 0: 
            done = True
        else:
            done = False

        # Apply reward and step information
        self.state = convert_index_to_state(len(self.cropNamesEN), action)
        self.state_index = action
        self.reward = reward

        info = {}
        
        # Return step information
        return self.state, self.reward, done, info
    
    # Render human readable information
    def render(self, mode='human'):
      log = ("Previous crop: " + str(self.initial_index) + " " + str(self.cropNamesEN[self.initial_index]) + "\tCurrent crop: " + str(self.state_index) + " " + str(self.cropNamesEN[self.state_index]) 
      + "\tSuitability: " + str(self.suitabilityMatrix[self.initial_index][self.state_index]) 
      + "\tCrop counter: " + str(self.cropCultivationCounterList[self.state_index]) + "/" + str(self.cropMaxCultivationTimesList[self.state_index])
      + "\tRow crop: " + str(self.cropRootCropList[self.initial_index]) + "-" +  str(self.cropRootCropList[self.state_index])
      #+ "\tPause violated: " + str(self.crop_break_rule_violated) +
      + "\tSoil: " + str(self.soilNitrogenList[self.state_index]) + " = " + str(self.soilNitrogenLevel)
      + "\tReward: " + str(self.reward))
      print(log)

    # Reset environment
    def reset(self):
        self.state_index= random.randint(0,len(self.cropNamesEN)-1)
        self.state = convert_index_to_state(len(self.cropNamesEN),self.state_index)
        self.soilNitrogenLevel = self.soilNitrogenLevelInit
        self.currentYield = 0
        self.cropRotationSequenceLength = self.cropRotationSequenceLengthStatic
        self.cropLastCultivationList = {
          0: -1,
          1: -1,
          2: -1,
          3: -1,
          4: -1,
          5: -1,
          6: -1,
          7: -1,
          8: -1,
          9: -1,
          10: -1,
          11: -1,
          12: -1,
          13: -1,
          14: -1,
          15: -1
        }
        self.cropLastCultivationList = {
          0: -1,
          1: -1,
          2: -1,
          3: -1,
          4: -1,
          5: -1,
          6: -1,
          7: -1,
          8: -1,
          9: -1,
          10: -1,
          11: -1,
          12: -1,
          13: -1,
          14: -1,
          15: -1,
          16: -1,
          17: -1,
          18: -1,
          19: -1,
          20: -1,
          21: -1,
          22: -1,
          23: -1,
          24: -1,
          25: -1
        }
        self.cropCultivationCounterList = {
          0: 0,
          1: 0,
          2: 0,
          3: 0,
          4: 0,
          5: 0,
          6: 0,
          7: 0,
          8: 0,
          9: 0,
          10: 0,
          11: 0,
          12: 0,
          13: 0,
          14: 0,
          15: 0
        }
        self.cropCultivationCounterList = {
          0: 0,
          1: 0,
          2: 0,
          3: 0,
          4: 0,
          5: 0,
          6: 0,
          7: 0,
          8: 0,
          9: 0,
          10: 0,
          11: 0,
          12: 0,
          13: 0,
          14: 0,
          15: 0,
          16: 0,
          17: 0,
          18: 0,
          19: 0,
          20: 0,
          21: 0,
          22: 0,
          23: 0,
          24: 0,
          25: 0
        }
        self.cropLastCultivationList[self.state_index] = 1

        # Store inital crop for later usage and increase the crop cultivation counter for the current crop by 1
        if self.cropIsLegumeList[self.state_index] == 1:
          for i in range(len(self.cropIsLegumeList)):
            if self.cropIsLegumeList[i] == 1:
              self.cropCultivationCounterList[i] += 1
              self.cropLastCultivationList[i] = 1
          else:
            self.cropCultivationCounterList[self.state_index] += 1
        
        return self.state