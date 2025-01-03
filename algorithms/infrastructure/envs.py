import gymnasium as gym
import numpy as np
import infrastructure.utils as utils


# What do we want the env_wrapper to do?
# Get action/env space
class EnvWrapper(gym.Wrapper):
    # instance of gym env
    def __init__(self, env, max_length=0, logger=None, logging_period=50):

        self.steps = 0
        self.max_length = max_length
        self.episode_count = -1

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.avg_reward = 0

        # reward trajectory
        self.trajectory = []

        # save all trajectories and every logging_period episodes calculate
        # stats
        self.saved_trajectories = []

        self.logging_period = logging_period
        self.logger = logger

        # counts the amount of times tensorboard logging happened
        self.log_step = 0


    """
        helper logging functions, etc.
    """


    def update_average_rew(self, new_reward):
        n = self.episode_count
        self.avg_reward *= (n-1)/n
        self.avg_reward += 1/n * new_reward

    def tensorboard_enabled(self):
        return self.logger is not None

    def log_statistics(self, trajectories):
        lengths = [len(traj) for traj in trajectories]
        total_rewards = [np.sum(traj) for traj in trajectories]
        mean_length = np.mean(lengths)
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)
        variance_reward = np.var(total_rewards)

        self.logger.log(mean_length, 'Average/Length', self.log_step)
        self.logger.log(mean_reward, 'Average/Reward', self.log_step)
        self.logger.log(std_reward, 'Reward/Standard Deviation', self.log_step)
        self.logger.log(max_reward, 'Reward/Max', self.log_step)
        self.logger.log(min_reward, 'Reward/Min', self.log_step)
        self.logger.log(variance_reward, 'Reward/Variance', self.log_step)
        
        self.log_step += 1


    """
        ENV INTERACTION
    """


    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.episode_count += 1

        # first reset -> don't log anything
        if self.episode_count == 0:
            return utils.to_torch(obs), info

        # Save this episode reward trajectory
        self.saved_trajectories.append(self.trajectory)

        # Update average per episode reward
        self.update_average_rew(np.sum(self.trajectory))

        if self.episode_count % self.logging_period == 0:
            print(f"Episode {self.episode_count}: avg per episode reward collected in", 
                  f"env - {self.avg_reward}. Last episode reward - {np.sum(self.trajectory)}")
            if self.tensorboard_enabled():
                self.log_statistics(self.saved_trajectories)
            self.saved_trajectories = []

        self.trajectory = []
        self.steps = 0
        return utils.to_torch(obs), info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        # truncate if max ep len reached
        if self.max_length and self.steps == self.max_length:
            truncated = True

        self.trajectory.append(reward)

        return utils.to_torch(obs), reward, terminated, truncated, info



