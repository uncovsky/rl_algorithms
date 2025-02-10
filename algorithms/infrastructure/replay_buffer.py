import infrastructure.utils as utils
import numpy as np
import torch


class RingBuffer:
    def __init__(self, capacity):
        self.idx = 0
        self.capacity = capacity

        # s, a, r, s', term
        self.transitions = []

    def insert(self, transition):
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.idx] = transition
            self.idx = ( self.idx + 1 ) % self.capacity


    def sample_indices(self, batch_size):
        return np.random.choice(len(self.transitions), size=batch_size)

    def get_transitions(self, indices):
        transitions = [ self.transitions[i] for i in indices ]

        # unpack into batches of states, actions, etc.
        res = list(zip(*transitions))

        # Stacking since the states are already tensors, just need
        # to stack them up along new dimension
        states = torch.stack(res[0]).to(utils.device)

        # Unsqueezing adds an extra dimension - batches, this is
        # used when gathering and multiplying w. rewards
        actions = utils.to_torch(np.array(res[1])).unsqueeze(1)
        rews = utils.to_torch(np.array(res[2],
            dtype=np.float32)).unsqueeze(1)
        succs = torch.stack(res[3]).to(utils.device)
        terms = utils.to_torch(np.array(res[4]))

        return states, actions, rews, succs, terms


    def sample(self, batch_size):
        if batch_size > len(self.transitions):
            return ([], [], [], [], [])

        indices = self.sample_indices(batch_size)
        self.get_transitions(indices)


    """
        Calculate return estimates for several indices
    """
    def calculate_n_step_return(self, policy, indices, gamma, n=5):
        res = torch.zeros(len(indices))

        for i in range(len(indices)):
            index = indices[i]
            discount = 1.0

            for shift in range(n):

                shifted_idx = ( index + shift ) % self.capacity
                s , a, r, succ, done = self.transitions[shifted_idx]

                res[i] += discount * r
                discount *= gamma

                if done:
                    break

                # bootstrap on final step
                if shift == n-1:
                    res[i] += discount * policy.value_nograd(succ).item()
        return res


    def calculate_gae(self, policy, indices, gamma, lbda):

        res = torch.zeros(len(indices))

        for i in range(len(indices)):
            index = indices[i]
            discount = 1.0
            end = False
            shift = 0

            while not end:
                shifted_idx = ( index + shift ) % self.capacity
                s, a, r, succ, done = self.transitions[shifted_idx]

                print(r)
                print(res[i])
                delta = policy.value_nograd(s) - r

                if not done:
                    delta += gamma * policy.value_nograd(succ)

                    res[i] += discount * delta.item()
                    discount *= gamma * lbda
                    shift += 1
                else:
                    print("joever", res[i])
                    end = True

        return res
        

    def __len__(self):
        return len(self.transitions)

