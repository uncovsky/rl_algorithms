import torch
import infrastructure.utils as utils
import torch.nn as nn

# Generic policy interface
class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, *args, **kwargs) -> int:
        raise NotImplementedError()

    # Should return the predicted Q-values for the given state
    def raw(self, state: int, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

def get_distribution(name, params):

    distributions = {
        "normal": torch.distributions.Normal,
        "uniform": torch.distributions.Uniform,
        "bernoulli": torch.distributions.Bernoulli,
        "poisson": torch.distributions.Poisson,
        "exponential": torch.distributions.Exponential,
        "categorical": torch.distributions.Categorical,
    }
    
    name = name.lower() 
    if name in distributions:
        if name == "categorical":
            return distributions[name](logits=params)
        return distributions[name](*params)
    else:
        raise ValueError(f"policies.py :: get_distribution - Unknown distribution name: '{name}'.")


class StochasticPolicy(Policy):
    def __init__(self, network, value_network, dist_str):

        """
            `dist_str` is a string that describes what distribution should be
            used, see `get_distribuiton above`
        """
        self.net = network
        self.value_net = value_network
        self.dist_str = dist_str


    def policy_params(self):
        return self.net.parameters()

    def value_params(self):
        return self.value_net.parameters()

    @torch.no_grad()
    def value_nograd(self, states):
        return self.value_net(states)
    
    def value(self, states):
        return self.value_net(states)

    @torch.no_grad
    def play(self, states):
        # Assume network returns mu, logstd
        dist = self.get_distribution(states)
        actions = dist.sample()
        return utils.to_numpy(actions)

    def logits(self, states):
        return self.net(states)

    # Get action distribution for a state/states
    def get_distribution(self, states):
        params = self.net(states)
        dist = get_distribution(self.dist_str, params)
        return dist
    
    # Get logprobs for actions at states
    def logprob(self, states, actions):
        dist = self.get_distribution(states)
        return dist.log_prob(actions)

"""
    Stochastic policy ~ Categorical distribution over a discrete space of
    actions, i.e. finitely many actions are available.
"""
class DiscretePolicy(StochasticPolicy):
    def __init__(self, network, value_network):
        super(DiscretePolicy, self).__init__(network, value_network, "categorical")

    def get_distribution(self, states):
        params = self.net(states)
        dist = get_distribution(self.dist_str, params)
        return dist

"""
    Gaussian policy over a continuous action space.
"""
class GaussianPolicy(StochasticPolicy):
    def __init__(self, network, value_network):
        super(GaussianPolicy, self).__init__(network, value_network, "normal")

    # assume the params are mu, logstd
    def get_distribution(self, states):
        params = self.net(states)
        size = params.size()

        if len(size) == 1:
            half = size[0] // 2
            mu, logstd = params[:half], params[half:]

        else:
            half = size[1] // 2
            mu, logstd = params[:, :half], params[:, half:]

        std = torch.exp(logstd)
        return torch.distributions.Normal(mu, std)

