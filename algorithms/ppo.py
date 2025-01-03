import infrastructure.utils as utils
from infrastructure.logger import Logger
from infrastructure.pg_utils import initialize_layer

import gymnasium as gym
import numpy as np

import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as F 

"""
    The familiar Policy/Trainer interface:
"""

def initialize_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, *args, **kwargs) -> int:
        raise NotImplementedError()

    # Should return the predicted Q-values for the given state
    def raw(self, state: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()



"""
    Every trainer features a logger class that is called during training.
"""
class Trainer:
    def __init__(self, env, logger=None, *args, **kwargs):
        self.env = env
        self.logger = logger

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma : float, steps : int, *args, **kwargs) -> Policy:
        raise NotImplementedError()


    def logging(self):
        return logger is not None
"""
    Networks used for Value and Policy network, feel free to modify their
    structure, activations, etc.
"""

class ValueNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ValueNet, self).__init__()
        self.layer1 = initialize_layer(nn.Linear(input_size, hidden_size))
        self.layer2 = initialize_layer(nn.Linear(hidden_size, hidden_size))
        self.layer3 = initialize_layer(nn.Linear(hidden_size, output_size),
                std=1.)

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x

    @torch.no_grad()
    def value(self, obs):
        return self(obs)


class PolicyNet(nn.Module):
    # input ~ dimensions of state space, output ~ action count (discrete envs)
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.layer1 = initialize_layer(nn.Linear(input_size, hidden_size))
        self.layer2 = initialize_layer(nn.Linear(hidden_size, hidden_size))
        self.layer3 = initialize_layer(nn.Linear(hidden_size, output_size),
                std=0.01)


    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        
        # returns logits
        return x

    @torch.no_grad()
    def play(self, obs):
        logits = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()



class PGPolicy(Policy):
    def __init__(self, net : PolicyNet, value_net : ValueNet):
        self.net = net
        self.value_net = value_net

    # Returns played action in state
    def play(self, state):
        return self.net.play(state)

    # Returns value
    def value(self, state):
        return self.value_net.value(state)

"""
    TRAINER
"""

class PGTrainer:

    def __init__(self, env, state_dim, num_actions, 
                 policy_lr=1e-3, value_lr=1e-3,
                 gae_lbda=0.99, ppo_clamp=0.2,
                 ppo_epochs=2, ppo_minibatches=4,
                 batch_size=10, output=True, validation_env=None):
        """
            env: The environment to train on
            state_dim: The dimension of the state space
            num_actions: The number of actions in the action space
            policy_lr: The learning rate for the policy network.
            value_lr: The learning rate for the value network. 
            gae_lbda: The GAE discounting parameter lambda
            batch_size: The batch size (num of episodes for each learning step) 
        """

        self.env = env
        self.batch_size = batch_size
        self.output = output
        self.validation_env = validation_env

        self.lbda = gae_lbda
        self.ppo_clamp = ppo_clamp
        self.ppo_epochs = ppo_epochs
        self.minibatches = ppo_minibatches

        self.policy_net = PolicyNet(state_dim, num_actions)
        self.value_net = ValueNet(state_dim, 1)
        self.value_loss = nn.MSELoss(reduction='mean')

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                lr=value_lr)


    """
        Calculating GAE and discounted rewards
    """

    def calculate_rew(self, rews, dones, gamma):

        G = 0
        res = torch.zeros(len(rews))

        for idx in range(len(rews) - 1, -1, -1):

            # Reset running reward
            if dones[idx]:
                G = 0

            G = rews[idx] + gamma * G
            res[idx] = G

        return res


    def calculate_gae(self, rews, states, dones, gamma):

        gae = 0
        res = torch.zeros(len(rews))

        for idx in range(len(rews) - 1, -1, -1):

            # reset gae counter on ends of episodes
            if dones[idx]:
                gae = 0

            # TD error, add bootstrap if not the last state in episode
            delta = rews[idx] - self.value_net.value(states[idx])
            
            # if not end of episode, add bootstrap term
            if not dones[idx]:
                delta += gamma * self.value_net.value(states[idx + 1])

            gae = delta + gamma * self.lbda * gae

            res[idx] = gae

        return res

    """
        Calculating logprobs and targets (adv + returns)
    """

    def get_logprobs(self, states, actions):
        return Categorical(logits=self.policy_net(states)).log_prob(actions).unsqueeze(-1)

    def calculate_targets(self, states, rewards, dones, gamma=0.99):

        advantages = self.calculate_gae(rewards, states, dones, gamma).unsqueeze(-1)
        returns = torch.zeros_like(advantages)

        with torch.no_grad():
            # Q = A + V
            returns = advantages + self.value_net(states)

        return returns, advantages

    """
        Policy update method
    """
    def ppo_update_policy_net(self, states, actions, advantages, returns):

        logprobs = self.get_logprobs(states, actions) 

        old_logprobs = logprobs.clone().detach()

        num_transitions = logprobs.shape[0]

        total_loss = torch.tensor([0.0], dtype=torch.float32)
        total_vloss = torch.tensor([0.0], dtype=torch.float32)
        vloss_fn = nn.HuberLoss(reduction='mean', delta=0.5)

        for epoch in range(self.ppo_epochs):

            indices = torch.randperm(num_transitions)
            batch_size = self.batch_size // self.minibatches

            for batch in range(self.minibatches):

                selected = indices[batch * batch_size: (batch+1) * batch_size]

                s_actions = actions[selected]
                s_adv = advantages[selected]
                s_returns = returns[selected]
                s_states = states[selected]
                s_logold = old_logprobs[selected]

                logprobs_new = self.get_logprobs(s_states, s_actions)

                ratio = torch.exp(logprobs_new - s_logold)

                """
                    Policy loss
                """
                loss1 = s_adv * ratio
                loss2 = torch.clamp(ratio, min=1-self.ppo_clamp,max=1+self.ppo_clamp) * s_adv
                loss = -torch.minimum(loss1, loss2).mean()

                total_loss += loss

                """
                    Value loss
                """

                values = self.value_net(s_states)
                value_loss = 0.5 * vloss_fn(values, s_returns)
                total_vloss += value_loss

                """
                    Optimization step
                """
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                value_loss.backward()
                loss.backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

        return total_loss, total_vloss

            

    def train(self, gamma, step_limit, seed=None):

        """
            Train the agent for number of episodes specified by `train_episodes`, 
            while using the supplied discount `gamma`.
        """

        train_steps = step_limit // self.batch_size
        self.env.reset(seed=seed)
        running_mean_rew = 0
        total_steps = 0

        for i in range(train_steps):

            policy = PGPolicy(self.policy_net, self.value_net)
            states, actions, rew, dones, info = collect_trajectories(self.env, policy, 
                                                                     self.batch_size,
                                                                     gamma, False)
            total_steps += len(states)
        

            rewm = info['rewm']
            lenm = info['lenm']

            if i == 0:
                running_mean_rew = rewm
            else:
                running_mean_rew = 0.9 * running_mean_rew + 0.1 * rewm

            state_tensor = torch.stack(states)
            returns, advantages = self.calculate_targets(state_tensor, rew, dones)
            action_tensor = torch.tensor(actions)
            logits = self.policy_net(state_tensor)

            policy_loss, value_loss = self.ppo_update_policy_net(state_tensor, action_tensor, advantages, returns)

            if self.output:
                returns_sel = []
                adv_sel = []
                for j in range(-1, len(returns) - 1):
                    if j == -1 or dones[j]:
                        returns_sel.append(returns[j+1].item())
                        adv_sel.append(advantages[j+1].item())
                print("Returns:", returns_sel)
                print("Advantages:", adv_sel)
                        

                print(f"Epoch {i+1}, undiscounted ep reward - {rewm}")
                print(f"             length of episodes     - {lenm}")
                print(f"             value loss             - {value_loss.item()}")
                print(f"             policy loss            - {policy_loss.item()}")
                print(f"             total steps            - {total_steps}")


                if i % 10 == 0 and self.validation_env is not None:
                    eval_policy(self.validation_env, "validation",
                          PGPolicy(self.policy_net, self.value_net), 2,
                          log=False)

        return PGPolicy(self.policy_net, self.value_net)



def collect_trajectories(env, policy, step_limit, gamma, bootstrap_trunc, episode_limit=0):

    """
    This is a helper function that collects a batch of episodes,
    totalling `step_limit` in steps. The last episode is truncated to
    accomodate for the given limit.
    

    You can use this during training to get the necessary data for learning.

        Returns several flattened tensors:

            1) States encountered
            2) Actions played
            3) Rewards collected
            4) Dones - Points of termination / truncation.

        
        Whenever done[i] is True, the following tensor entries from i+1
        belong to the following episode (up to the next point where done[j] is
        True again).

        If `bootstrap_trunc` is true and an episode is truncated at timestep i,
        gamma * policy.value(next_state) is added to rewards[i]. Note that if you
        are not utilizing a value network, this should be turned off.

    You can modify this function as you like or even erase it.

    """

    states, actions, rewards, dones = [], [], [], []

    ep_rewards = []
    ep_lengths = []

    steps = 0
    ep_length = 0
    ep_reward = 0
    
    while steps < step_limit:

            obs, _ = env.reset()
            obs = utils.to_torch(obs)
            done = False
            disc = 1

            while not done:

                # remember to cast observations to tensors for your models
                action = policy.play(obs)
                states.append(obs)
                actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)

                ep_reward += reward 
                ep_length += 1

                steps += 1
                obs = utils.to_torch(obs)

                truncated = truncated or steps == step_limit or ep_length == episode_limit
        
                # Optionally bootstrap on truncation
                if truncated and bootstrap_trunc:
                    bootstrap = utils.to_numpy(gamma * policy.value(obs))[0]
                    reward += bootstrap

                rewards.append(reward)

                if terminated or truncated:
                    done = True 
                    ep_rewards.append(ep_reward)
                    ep_lengths.append(ep_length)
                    ep_length = 0
                    ep_reward = 0

                dones.append(done)
    
    # Don't count the lsat episode
    info = { 
            "lenm" : np.mean(ep_lengths[:-1]),
            "rewm" : np.mean(ep_rewards[:-1]),
    }

    return states, actions, rewards, dones, info


"""
    Wrapper for CarRacing
"""
class InputPreprocessor(gym.ObservationWrapper):
    def __init__(self, env):
        super(InputPreprocessor, self).__init__(env)

        self.center = (70, 48)
        self.rads = [45, 30, 15]
        self.angles = [-10/12, -11/12, -12/12, -13/12, -14/12]

        # Set observation space to have size len(rads) * len(angles)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(self.rads) * len(self.angles),), dtype=np.float32)

    def observation(self, obs):
        angular_coordinates = [(r, angle) for r in self.rads for angle in self.angles]

        outputs = np.zeros(len(angular_coordinates), dtype=np.float32)
        for i, (r, a) in enumerate(angular_coordinates):
            x = int(self.center[0] + r * np.cos(a*np.pi))
            y = int(self.center[1] + r * np.sin(a*np.pi))

            outputs[i] = obs[x, y, 1] > 180
        
        return outputs

def eval_policy(env, env_name, policy, eval_eps, log=True):

    traj_rewards = torch.zeros(eval_eps)
    for i in range(eval_eps):

        done = False
        state, _ = env.reset()
        state = utils.to_torch(state)

        total_rew = 0

        while not done:

            state, reward, terminated, truncated, _ = env.step(policy.play(state))
            state = utils.to_torch(state)
            total_rew += reward

            if terminated or truncated:
                done = True 
                traj_rewards[i] = total_rew

                #print(f"Eval Episode {i+1} - reward {total_rew}.")

    print(f"Mean reward - {torch.mean(traj_rewards)}, ({torch.std(traj_rewards)})")

# Single params for all benchmarks, just double batch size on lander
params = {
        'policy_lr' : 0.0003,
        'value_lr' : 0.0003,
        'ppo_clamp' : 0.2,
        'ppo_epochs': 10,
        'ppo_minibatches' : 32,
        'gamma' : 0.99,
        'gae_lbda' : 0.95,
        'batch_size' : 2048,
        'output' : False,
}


def train_cartpole(env, train_steps, gamma) -> PGPolicy:
    state_dim, num_actions = utils.get_env_dimensions(env)
    trainer = PGTrainer(env, state_dim, num_actions,
                        policy_lr=params["policy_lr"],
                        value_lr=params["value_lr"], 
                        ppo_clamp=params['ppo_clamp'],
                        ppo_epochs=params['ppo_epochs'],
                        ppo_minibatches=params['ppo_minibatches'],
                        gae_lbda = params["gae_lbda"],
                        batch_size=params["batch_size"],
                        output=params['output'],
                        validation_env=None,
                        )
    # Train the agent on 1000 steps.
    pol = trainer.train(params["gamma"], steps)
    return pol


def train_acrobot(env, train_steps, gamma) -> PGPolicy:
    state_dim, num_actions = utils.get_env_dimensions(env)
    trainer = PGTrainer(env, state_dim, num_actions,
                        policy_lr=params["policy_lr"],
                        value_lr=params["value_lr"], 
                        ppo_clamp=params['ppo_clamp'],
                        ppo_epochs=params['ppo_epochs'],
                        ppo_minibatches=params['ppo_minibatches'],
                        gae_lbda = params["gae_lbda"],
                        batch_size=params["batch_size"],
                        output=params['output'],
                        validation_env=None,
                        )
    # Train the agent on 1000 steps.
    pol = trainer.train(params["gamma"], steps)
    return pol

def train_lunarlander(env, train_steps, gamma) -> PGPolicy:
    state_dim, num_actions = utils.get_env_dimensions(env)
    trainer = PGTrainer(env, state_dim, num_actions,
                        policy_lr=params["policy_lr"],
                        value_lr=params["value_lr"], 
                        ppo_clamp=params['ppo_clamp'],
                        ppo_epochs=params['ppo_epochs'],
                        ppo_minibatches=params['ppo_minibatches'],
                        gae_lbda = params["gae_lbda"],
                        batch_size=4096,
                        output=params['output'],
                        validation_env=None,
                        )
    # Train the agent on 1000 steps.
    pol = trainer.train(params["gamma"], steps)
    return pol


"""
    CarRacing is a challenging environment for you to try to solve.
"""

RACING_CONTINUOUS = False


def train_carracing(env, train_steps, gamma) -> PGPolicy:
    """
        As the observations are 96x96 RGB images you can either use a
        convolutional neural network, or you have to flatten the observations.

        You can use gymnasium wrappers to achieve the second goal:
    """
    env = InputPreprocessor(env)

    """
        The episodes in this environment can be very long, you can also limit
        their length by using another wrapper.

        Wrappers can be applied sequentially like so:
    """

    state_dim, num_actions = utils.get_env_dimensions(env)
    trainer = PGTrainer(env, state_dim, num_actions,
                        policy_lr=params["policy_lr"],
                        value_lr=params["value_lr"], 
                        ppo_clamp=params['ppo_clamp'],
                        ppo_epochs=params['ppo_epochs'],
                        ppo_minibatches=params['ppo_minibatches'],
                        gae_lbda = params["gae_lbda"],
                        batch_size=params["batch_size"],
                        output=params['output'],
                        validation_env=None,
                        )
    # Train the agent on 1000 steps.
    pol = trainer.train(params["gamma"], steps)
    return pol


def wrap_carracing(env):
    env = InputPreprocessor(env)
    return env

def wrap_cartpole(env):
    return env

def wrap_acrobot(env):
    return env

def wrap_lunarlander(env):
    return env

