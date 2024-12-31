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

    def sample(self, batch_size):

        if batch_size > len(self.transitions):
            return ([], [], [], [], [])


        indices = np.random.choice(len(self.transitions), size=batch_size)

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

    def __len__(self):
        return len(self.transitions)

