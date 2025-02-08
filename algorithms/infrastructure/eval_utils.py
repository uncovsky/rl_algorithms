import gymnasium as gym
import torch
import infrastructure.utils as utils


def eval_policy(env, policy, eval_eps=100):
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

    env.reset()

    return traj_rewards
