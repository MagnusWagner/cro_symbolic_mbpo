# SET LENGTH HERE
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import random
from .data.data import cropNamesDE, cropNamesEN, suitabilityMatrix_Kolbe, suitabilityMatrix_NDVI, soilNitrogenList, cropYieldList, cropCultivationBreakList, cropLastCultivationList, cropMaxCultivationTimesList, cropCultivationCounterList, cropRootCropList, cropIsLegumeList
cropLastCultivationList

def convert_state_to_index(state):
   return np.argmax(state)

def convert_index_to_state(length, index):
  state = np.zeros(length)
  state[index] = 1.0
  return state

class CropRotationEnv(Env):
    def __init__(self, seed = 42, seq_len = 7, NMin = 75, NMax = 76, neg_reward = -5000):
        random.seed(seed)
        self.cropRotationSequenceLengthStatic = seq_len
        self.NMin = NMin
        self.NMax = NMax
        self.negativeReward = neg_reward
        # German names of crops used in the crop rotation
        self.cropNamesDE = cropNamesDE
        
        # English names of crops used in the crop rotation
        self.cropNamesEN = cropNamesEN
        
        # Successor crop suitability matrix, -1: non-suitable combination, 1: good combination, 2: very good combination
        # row = current crop, column = next crop
        # Kolbe Matrix
        #self.suitabilityMatrix = suitabilityMatrix_Kolbe
        # Suitability Matrix derived from real NDVI effects in Lower Austria from 2018 to 2021. -1, 1 and 2 
        self.suitabilityMatrix = suitabilityMatrix_NDVI
        # Nitrogen balance in soil after crop is harvested
        self.soilNitrogenList = soilNitrogenList
        
        # Yield 
        self.cropYieldList = cropYieldList

        # Cultivation breaks
        self.cropCultivationBreakList = cropCultivationBreakList
        
        # Last cultivation
        self.cropLastCultivationList = cropLastCultivationList
        
        # Maximum cultivation times within a crop rotation sequence
        # Legumes are set to 2 because of upper 20% limit within a 10-step sequence
        self.cropMaxCultivationTimesList = cropMaxCultivationTimesList

        # Cultivation counter within a crop rotation sequence
        self.cropCultivationCounterList = cropCultivationCounterList

        # Indicator if crop is a root crop; 0: no root crop, 1: root crop
        self.cropRootCropList = cropRootCropList

        # Indicator if crop is a legume; 0: no legume, 1: legume
        self.cropIsLegumeList = cropIsLegumeList
        


        # Action and observation space are defined by crops in cropNames
        self.action_space = Discrete(len(self.cropNamesEN)) # Discrete actions: Select a crop to grow
        self.observation_space = Box(low=np.zeros(len(self.cropNamesEN)), high=np.ones(len(self.cropNamesEN)), shape = (len(self.cropNamesEN),), dtype=np.int16)
        
        # Initial crop of the crop sequence is selected randomly; initial state is stored for later usage
        self.state_index = random.randint(0,len(self.cropNamesEN)-1)
        self.initial_index = self.state_index
        self.state = convert_index_to_state(len(self.cropNamesEN),self.state_index)
        self.initial = self.state
        

        # Soil is initialized with a nitrogen level of NMin; current yield and reward are set to 0; crop rotation sequence length set to 10
        self.soilNitrogenLevelInit = random.uniform(a = self.NMin, b = self.NMax)
        self.soilNitrogenLevel = self.soilNitrogenLevelInit
        self.currentYield = 0
        self.reward = 0
        
        
        # Set the last cultivation step for the current crop to 1
        self.cropLastCultivationList[self.state_index] = 1
    
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