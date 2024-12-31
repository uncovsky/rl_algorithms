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
        "normal": torch.distributions.multivariate_normal.MultivariateNormal,
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

    @torch.no_grad()
    def value_nograd(self, states):
        return self.value_net(states)
    
    def value(self, states):
        return self.value_net(states)

    @torch.no_grad
    def play(self, state):
        # Assume network returns mu, logstd
        dist = self.get_distribution(state)
        action = dist.sample()
        return utils.to_numpy(action)

    # Get action distribution for a state/states
    def get_distribution(self, states):
        params = self.net(states)
        dist = get_distribution(self.dist_str, params)
        return dist
    
    # Get logprobs for actions at states
    def logprob(self, states, actions):
        dist = self.get_distribution(states)
        return dist.log_prob(actions).unsqueeze(-1)


class DiscretePolicy(StochasticPolicy):
    def __init__(self, network, value_network):
        super(DiscretePolicy, self).__init__(network, value_network, "categorical")

    def get_distribution(self, states):
        params = self.net(states)
        dist = get_distribution(self.dist_str, params)
        return dist

class GaussianPolicy(StochasticPolicy):
    def __init__(self, network, value_network):
        super(GaussianPolicy, self).__init__(network, value_network, "normal")

    # assume the params are mu, logstd
    def get_distribution(self, states):
        params = self.net(states)
        half = len(params) // 2

        mu, logstd = params[:half], params[half:]
        std = torch.exp(logstd)
        # Diagonal Covariance
        params = [ mu, torch.diag(std) ]

        dist = get_distribution(self.dist_str, params)

        return dist

