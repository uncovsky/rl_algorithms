import infrastructure.utils as utils
from infrastructure.logger import Logger

from nn_utils import make_mlp
import numpy as np

from pg_utils import initialize_layer
from policies import GaussianPolicy, DiscretePolicy

from replay_buffer import RingBuffer

import torch
from torch.optim import Adam

"""
    Soft Actor-Critic
"""

class SAC_Trainer:

    def __init__(self, env, input_size, output_size, hidden_size=256, #net structure
                 learning_rate = 3e-4, batch_size=256, activation="tanh",
                 entropy_multiplier=1, tau=0.005, buffer_size=100000):

        self.env = env

        # Network descriptions, 2 layers
        value_desc = [ (activation, input_size, hidden_size), 
                       ("id", hidden_size, 1) ]
        policy_desc = [ (activation, input_size, hidden_size), 
                        ("id", hidden_size, 2 * output_size) ]

        # Hyperparameters 
        self.tau = tau
        self.alpha = entropy_multiplier

        # Policy network
        self.policy_net = make_mlp(policy_desc)
        
        # Two Q-value nets for reduced bias
        self.q1_net = make_mlp(policy_desc)
        self.q2_net = make_mlp(policy_desc)

        # Value net and target, perhaps try later?
        self.value_net = make_mlp(value_desc)
        self.value_target_net = make_mlp(value_desc)

        self.initialize_networks()

        # Copy params to target
        self.value_target_net.load_state_dict(self.value_net.state_dict())

        # Size of buffer and minibatches
        self.buffer = RingBuffer(buffer_size)
        self.batch_size = batch_size

        # Optimizers
        self.value_opt = Adam(self.value_net.parameters(), lr=learning_rate)
        self.policy_opt = Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Two Q function optimizers
        self.q2_opt = Adam(self.q1_net.parameters(), lr=learning_rate)
        self.q1_opt = Adam(self.q2_net.parameters(), lr=learning_rate)



    # Initializes layers with appropriate random weights
    def initialize_networks(self):

        # Init policy net
        layers = self.policy_net[::2]
        for layer in layers[:-1]:
            initialize_layer(layer)
        initialize_layer(layers[-1], std=0.01)
        
        # Init value net
        layers = self.value_net[::2]
        for layer in layers[:-1]:
            initialize_layer(layer)
        initialize_layer(layers[-1], std=1.)

        self.value_target_net.to(utils.device)
        self.value_net.to(utils.device)
        self.policy_net.to(utils.device)
        self.q1_net.to(utils.device)
        self.q2_net.to(utils.device)


    def get_policy(self):
        return GaussianPolicy(self.policy_net, self.value_net)


    def soft_update_target(self):
        params = self.value_target_net.state_dict()
        value_params = self.value_net.state_dict()

        for key in params.keys():
            params[key] *= (1 - self.tau)
            params[key] += self.tau * value_params[key]
        
    def train(self, step_limit, gamma=0.99):

        self.env.reset()
        steps = 0

        while steps < step_limit:

                obs, _ = self.env.reset()
                obs = utils.to_torch(obs)
                done = False

                while not done:

                    """ 
                        Take a step and record into replay buffer.
                    """
                    policy = self.get_policy()
                    action = policy.play(obs)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    next_obs = utils.to_torch(next_obs)

                    transition = (obs, action, reward, next_obs, terminated)
                    self.buffer.insert(transition)
                    obs = next_obs

                    # Update policy net, value net, Q networks
                    self.update(gamma)

                    # Update value target network
                    self.soft_update_target()

                    truncated = truncated or steps == step_limit

                    if terminated or truncated:
                        done = True 

    def calculate_targets(self, states, rewards, succs, dones, gamma):

        bootstrap = torch.min(self.q1_net(succs), self.q2_net(succs))

    def update(self, gamma):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = self.buffer.sample(self.batch_size)

    
        states = torch.stack(batch[0], dim=0)
        succs = torch.stack(batch[3], dim=0)

        actions = torch.tensor(np.array(batch[1]))

        print(actions)
        rewards = torch.tensor(batch[2]).unsqueeze(-1)

        dones = torch.tensor(batch[4]).unsqueeze(-1)

        # Calculate targets
        self.calculate_targets(states, rewards, succs, dones, gamma)

    def loss_fn(self):
        pass

