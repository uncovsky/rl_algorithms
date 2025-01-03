from sac import SAC_Trainer

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, DtypeObservation

import infrastructure.utils as utils
import numpy as np



env = gym.make("CarRacing-v3", continuous=True)
env.observation_space.shape
env = DtypeObservation(FlattenObservation(env), np.float32)


state_dim, action_dim = utils.get_env_dimensions(env)


trainer = SAC_Trainer(env, state_dim, action_dim, batch_size=12)
trainer.train(1000)

