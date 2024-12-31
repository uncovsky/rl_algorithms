# DQN

import gymnasium as gym
import infrastructure.utils as utils
import infrastructure.envs as envs
from infrastructure.logger import Logger

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import RingBuffer

class DQNNet(nn.Module):
    # input ~ dimensions of state space, output ~ action count (discrete envs)
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        return x

    @torch.no_grad()
    def play(self, obs, eps=0.0):
        qvals = self(obs)
        if np.random.rand() <= eps:
            return np.random.choice(len(qvals))
        x = torch.argmax(qvals)

        # cast from tensor to int so gym does not complain
        return int(x)

def evaluate_dqn(env, agent, params, steps=100000):

    agent.eval()
    avg_reward, returns, values = [], [], []
    gamma = params['gamma']

    observation, info = env.reset()
    step = 0

    while step < steps:

        done = False
        ep_rewards = []

        while not done and step < steps:

            # append q value of initial state
            with torch.no_grad():
                values.append(utils.to_numpy(agent(observation).max(dim=-1).values))

            action = agent.play(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            step += 1

            ep_rewards.append(reward)


            if terminated or truncated:

                G = 0
                for rew in ep_rewards[::-1]:
                    G = rew + gamma * G
                    # don't care about order here
                    returns.append(G)

                avg_reward.append(np.sum(ep_rewards))
                done = True 
                observation, info = env.reset()

    agent.train()
    return np.mean(avg_reward), np.mean(returns), np.mean(values)
                        

    

class DQNTrainer:

    def __init__(self, buf_capacity, env, eval_env, hidden_size=168, lr=1e-4,
            log_name=""):
        input_size, output_size = utils.get_env_dimensions(env)

        self.value_net = DQNNet(input_size, hidden_size, output_size)
        self.target_net = DQNNet(input_size, hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.logger = Logger("results/DQN/" + log_name)

        utils.use_cpu()

        self.value_net.to(utils.device)
        self.target_net.to(utils.device)
        self.target_net.load_state_dict(self.value_net.state_dict())

        self.buffer = RingBuffer(buf_capacity)
        self.loss = nn.MSELoss(reduction='mean')
        self.env = env
        self.eval_env = eval_env


    def soft_update_target_net(self, tau):

        params = self.target_net.state_dict()
        value_params = self.value_net.state_dict()

        # Polyak averaging of target parameters
        for key in params.keys():
            params[key] *= (1 - tau)
            params[key] += tau * value_params[key]

        self.target_net.load_state_dict(params)
        


    def train(self, steps, params):

        env = self.env

        gamma = params['gamma']
        log_period = params['log_period']
        eps = params['eps_beg']
        batch_size = params['batch_size']
        eps_end = params['eps_end']
        tau = params['tau']

        eps_step = (eps - eps_end) / ( steps // 2 )
        self.value_net.train()

        step = 0
        ep = 0
        observation, info = env.reset(seed=params['seed'])

        while step < steps:

            done = False

            while not done and step < steps:

                """
                    SAMPLE FROM ENV
                """

                step += 1
                action = self.value_net.play(observation, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.buffer.insert((observation, action, reward, next_state, terminated))
                observation = next_state

                """
                    SAMPLE MINIBATCH
                """
                if len(self.buffer) >= batch_size:

                    state_batch, action_batch, reward_batch, succ_batch, term_batch = self.buffer.sample(batch_size)

                    """
                        LEARNING BRANCH
                    """
                    preds = self.value_net(state_batch).gather(1, action_batch)

                    # Only add the bootstrap gamma * succ to nonterminal states
                    with torch.no_grad():
                        indices = (term_batch == 0).nonzero(as_tuple=True)[0]
                        targets = torch.ones(size=(batch_size,1), device=utils.device) * reward_batch
                        succ_indices = self.value_net(succ_batch[indices]).argmax(dim=1, keepdim=True)
                        targets[indices] += gamma * self.target_net(succ_batch[indices]).gather(1,succ_indices)

                    """
                        OPTIMIZATION STEP + LOSS CALCULATION
                    """
                    self.optimizer.zero_grad()
                    loss = self.loss(preds, targets)
                    loss.backward()
                    self.optimizer.step()
                    self.soft_update_target_net(tau)


                """
                    LOGGING/EVAL EVERY `log_period` steps

                    if step % log_period == 0:
                        rew, ret, val = evaluate_dqn(self.eval_env, self.value_net, params, steps=10000)

                        data = {
                            "episode_reward" : rew,
                            "avg_disc_returns" : ret,
                            "avg_q_values" : val
                        }

                        self.logger.write(data, step)
                """


                if eps > eps_end: 
                    eps -= eps_step

                if terminated or truncated:
                    done = True 
                    observation, info = env.reset()



if __name__ == "__main__":
    
    utils.list_devices()
    utils.init_device(0)

    seed = None

    env_names = ["CartPole-v1"]
    steps_dict = {
            "CartPole-v1": 30000,
            "Acrobot-v1" : 30000,
            "LunarLander-v2" : 100000,
            }
    reps = 1

    for env_name in env_names:
        for i in range(reps):

            env = gym.make(env_name, render_mode=None)
            eval_env = gym.make(env_name, render_mode=None)
            render_env = gym.make(env_name, render_mode='human')

            wrapped_env = envs.EnvWrapper(env, logger=None, max_length=0,
                    logging_period=20)
            wrapped_eval_env = envs.EnvWrapper(eval_env, logger=None,
                    max_length=0,logging_period=20)
            render_env = envs.EnvWrapper(render_env, logger=None, max_length=0, logging_period=1)
            input_size, output_size = utils.get_env_dimensions(env)

            """
            params = {
                    'lr' : 1e-4,
                    'gamma' : 0.99,
                    'tau' : 0.01,
                    'eps_beg' : 0.9,
                    'eps_end' : 0.05,
                    'log_period' : 10000,
                    'batch_size' : 64,
                    'seed' : seed,
            }
            """

            params = {
                    'lr' : 1e-4,
                    'gamma' : 0.99,
                    'tau' : 0.005,
                    'eps_beg' : 1.0,
                    'eps_end' : 0.1,
                    'log_period' : 1000,
                    'batch_size' : 32,
                    'seed' : seed,
            }

            steps = steps_dict[env_name]
            buf_size = steps // 10
            agent = DQNTrainer(buf_size, wrapped_env, wrapped_eval_env,
                    lr=params['lr'], log_name=f"{env_name}_{i}")
            agent.train(steps, params=params)

            print("Evaluation of the agent:\n")
            evaluate_dqn(render_env, agent.value_net, params, 10000)

