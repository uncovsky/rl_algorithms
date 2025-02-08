import infrastructure.utils as utils
import torch

class BC_Trainer:

    def __init__(self, dataset, input_policy, lr=1e-2, episode_batch_size=10, validation_cb=None):
        self.dataset = dataset
        self.lr = lr
        self.bs = episode_batch_size
        self.optimizer = torch.optim.Adam(input_policy.policy_params(), lr=lr)
        self.validation_cb = validation_cb
        self.policy = input_policy
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def step(self, states, actions):
        logits = self.policy.logits(states)
        self.optimizer.zero_grad()
        loss = self.loss_fn(logits, actions)
        loss.backward()
        self.optimizer.step()

    def train(self, epochs):

        for epoch in range(epochs):

            eps = self.dataset.sample_episodes(self.bs)

            action_batch = torch.tensor([])
            state_batch = torch.tensor([])
            for episode in eps:
                action_batch = torch.cat((action_batch, utils.to_torch(episode.actions)))
                state_batch = torch.cat((state_batch, utils.to_torch(episode.observations[:-1])))
        
            action_batch = action_batch.long()
            self.step(state_batch, action_batch)


            if self.validation_cb and epoch % 10 == 0:
                self.validation_cb(self.policy)
                

        return self.policy 


