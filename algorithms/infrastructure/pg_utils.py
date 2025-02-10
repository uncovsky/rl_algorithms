import numpy as np
import gymnasium as gym
import infrastructure.utils as utils
import torch


"""
    A set of utility functions useful for PG-like algorithms that rely on
    collecting trajectories / estimating MC/GAE returns.
"""


def calculate_rew(rews, dones, gamma):
    """
        Calculate MC estimate of returns for a set of trajectories.

        `rews` and `dones` are flattened shape (N, ) tensors with data pertaining to
        possibly multiple trajectories.

        returns `result` - shape (N, ) tensor with each cell corresponding to
        MC return estimate of the given state.
    """

    G = 0
    res = torch.zeros(len(rews))

    for idx in range(len(rews) - 1, -1, -1):

        # Reset running reward
        if dones[idx]:
            G = 0

        G = rews[idx] + gamma * G
        res[idx] = G

    return res


def calculate_gae(policy, rews, states, dones, gamma, gae_lambda=0.95):

    gae = 0
    res = torch.zeros(len(rews))

    for idx in range(len(rews) - 1, -1, -1):

        # reset gae counter on ends of episodes
        if dones[idx]:
            gae = 0

        # TD error, add bootstrap if not the last state in episode
        delta = rews[idx] - policy.value(states[idx])
        
        # if not end of episode, add bootstrap term
        if not dones[idx]:
            delta += gamma * policy.value(states[idx + 1])

        gae = delta + gamma * gae_lambda * gae

        res[idx] = gae

    return res


"""
    Collect trajectories from gym environment - limited by `step_limit`,
    potentially bootstrap on truncation.
"""

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
    Collect full episodes from gym env, no preemptive truncation
"""
def collect_full_episodes(env, policy, episode_count=10, gamma=0.99):

    states, actions, rewards, dones = [], [], [], []

    ep_rewards = []
    ep_lengths = []

    steps = 0
    ep_length = 0
    ep_reward = 0
    
    for i in range(episode_count):

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

            ep_reward += disc * reward 
            disc *= gamma
            ep_length += 1

            steps += 1
            obs = utils.to_torch(obs)

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
            "lenm" : np.mean(ep_lengths),
            "rewm" : np.mean(ep_rewards),
    }

    return states, actions, rewards, dones, info

