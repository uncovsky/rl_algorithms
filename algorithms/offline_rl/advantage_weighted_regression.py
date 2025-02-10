from copy import deepcopy
import infrastructure.utils as utils
from infrastructure.replay_buffer import RingBuffer
from infrastructure.pg_utils import collect_trajectories, calculate_gae
import torch


class AWR_Trainer:

    def __init__(self, env, input_policy, lr=1e-3, value_lr=2e-3, episode_batch_size=10, 
                       validation_cb=None, temp=0.5, weight_clip=20.0,
                        buffer_size = 50000, online_steps=100,
                        online_batch_size=64, 
                        value_grad_steps=1,
                        policy_grad_steps=2,
                       ):
        """
            Trainer for Advantage weighted regression - algorithm introduced in
                https://arxiv.org/abs/1910.00177.

            Supports two modes - 

            a) fully offline policy extraction from a learned
            value function - input_policy needs to have a learned value
            function on env dataset ~ either via IQL / CQL / sarsa.
            
            b) Reinforcement learning utilizing AWR as a supervised learning
            subroutine for policy improvement step.

            
            `validation_cb` specifies a callback to be used during training
            i.e. to estimate return of current policy pi_t, log, or print debug
            info


            `temp` is the tempearature parameter that weighs the advantage
            coefficients during training. Corresponds to "\beta" in the
            original paper.  Essentially, beta is the lagrange multiplier
            corresponding to the KL-divergence constraint on the updated policy
            (much like in PPO/TRPO). As beta approaches zero the update will
            shift more weight to actions w. higher advantage much like a greedy
            policy update.

            `weight_clip` - advantage weights are clipped to a range [0,
            weight_clip] for numerical stability reasons


            `buffer_size` is the size of replay buffer used during on-line
            training, i.e. scenario b) above
            
            `online_steps` is the number of interaction steps between a
            gradient step each epoch

            `online_batch_size` is the number of samples from replay buffer
            used for training each epoch.

            `policy/value_grad_steps` denotes the number of policy/value
            updates made each iteration

        """

        self.env = env
        self.lr = lr
        self.bs = episode_batch_size

        self.optimizer = torch.optim.Adam(input_policy.policy_params(), lr=lr)
        self.value_optimizer = torch.optim.Adam(input_policy.value_params(), lr=value_lr)

        self.validation_cb = validation_cb
        self.policy = input_policy

        self.buffer = RingBuffer(buffer_size)
        self.temp = temp

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.value_loss_fn = torch.nn.MSELoss(reduction='mean')

        self.online_bs = online_batch_size
        self.online_steps = online_steps

        self.wc = weight_clip

        self.value_grad_steps = value_grad_steps
        self.policy_grad_steps = policy_grad_steps


    def policy_loss(self, states, actions, advantage):
        logprobs = self.policy.logprob(states, actions)
        weight = torch.clamp(torch.exp(1/self.temp * advantage), max=self.wc)
        policy_loss = -1 * torch.mean(weight * logprobs)

        return policy_loss

    def value_loss(self, states, returns):
        values = self.policy.value(states)
        return self.value_loss_fn(values, returns)


    """
        Assumes `self.env` is a instance of gym environment, alternates between
        phases of training value function and policy improvement via AWR.
    """

    def train_online(self, gamma, epochs):

        for epoch in range(epochs):

            # Collect the data
            states, actions, rewards, dones, _ = collect_trajectories(self.env,
                    self.policy, self.online_steps, gamma, bootstrap_trunc=True)

            # Insert the transitions into buffer
            for i in range(self.online_steps):
                successor = states[i+1] if not dones[i] else states[i]
                self.buffer.insert((states[i], actions[i], rewards[i], successor, dones[i]))


            total_vloss = 0
            total_ploss = 0

            # Copy policy to use a target net here basically
            target_policy = deepcopy(self.policy)

            for train_iter in range(self.value_grad_steps):

                # Sample data from buffer, first calculate TD(lambda) returns and advantages
                indices = self.buffer.sample_indices(self.online_bs)

                # Use value net from previous iteration to calculate returns
                returns = self.buffer.calculate_n_step_return(target_policy, indices, gamma).unsqueeze(-1)
                states, actions, rewards, succs, dones = self.buffer.get_transitions(indices)

                # Take SGD step on value.
                value_loss = self.value_loss(states, returns)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                total_vloss += value_loss.item()

            for train_iter in range(self.policy_grad_steps):

                # Sample data from buffer, first calculate TD(lambda) returns and advantages
                indices = self.buffer.sample_indices(self.online_bs)
                returns = self.buffer.calculate_n_step_return(target_policy, indices, gamma).unsqueeze(-1)
                states, actions, rewards, succs, dones = self.buffer.get_transitions(indices)

                advantages = returns - self.policy.value_nograd(states)

                # Normalize advantages
                std = torch.std(advantages, dim=0)
                mean = torch.mean(advantages, dim=0)

                advantages = ( advantages - mean ) / ( std + 1e-8 )

                policy_loss = self.policy_loss(states, actions, advantages)

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

                total_ploss += policy_loss.item()

                log_data = {
                    "vloss" : total_vloss,
                    "ploss" : total_ploss 
                }

            print(total_ploss)
            print(total_vloss)
            print("ITER ")
            if self.validation_cb:
                self.validation_cb(self.policy, epoch, log_data)

    """
        Assumes `self.policy` contains a trained value function, trains policy
        net via advantage weighted regression
    """
    def train_offline(self, gamma, epochs):

        for epoch in range(epochs):

            eps = self.env.sample_episodes(self.bs)

            action_list = []
            state_list = []
            succ_list = []
            reward_list = []

            dones_list = []

            for episode in eps:
                action_list.append(utils.to_torch(episode.actions))
                state_list.append(utils.to_torch(episode.observations[:-1]))
                succ_list.append(utils.to_torch(episode.observations[1:]))
                reward_list.append(utils.to_torch(episode.rewards))
                dones_list.append(utils.to_torch(episode.terminations))



            action_batch = torch.cat(action_list, dim=0)
            state_batch = torch.cat(state_list, dim=0)
            succ_batch = torch.cat(succ_list, dim=0)
            reward_batch = torch.cat(reward_list, dim=0)
            dones_batch = torch.cat(dones_list, dim=0)

            action_batch = action_batch.long()
            advantages = torch.zeros(len(reward_batch))

            indices = (dones_batch == 0).nonzero(as_tuple=True)[0]

            # Just use TD(0) for advantage estimates
            advantages = reward_batch + gamma * indices * self.policy.value_nograd(succ_batch)
            policy_loss = self.policy_loss(state_batch, action_batch, advantages)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if self.validation_cb:
                self.validation_cb(self.policy, epoch)
                

        return self.policy 



