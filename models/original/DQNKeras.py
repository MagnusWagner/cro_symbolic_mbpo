from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy
import tensorflow
# import tensorflow.keras as keras



class DQNKeras():
    def __init__(self, env, seed = 42):  
        self.states = env.observation_space.shape
        self.actions = env.action_space.n

    def build_model(self):
        try:
            model
        except NameError:
            model = tensorflow.keras.models.Sequential()
        del model
        model = tensorflow.keras.models.Sequential()    
        model.add(tensorflow.keras.layers.Dense(24, activation='relu', input_shape=self.states))
        model.add(tensorflow.keras.layers.Dense(24, activation='relu'))
        model.add(tensorflow.keras.layers.Dense(self.actions, activation='linear'))
        return model
    
    def build_agent(self, model = None ):
        if not model:
            model = self.build_model() 
        memory = SequentialMemory(limit=100000, window_length=1)
        dqn = DQNAgent(model=model, memory=memory, policy=MaxBoltzmannQPolicy(), test_policy=MaxBoltzmannQPolicy(), enable_double_dqn=True, nb_actions=actions, nb_steps_warmup=1000, gamma=0.90, target_model_update=1e-3)
        return dqn