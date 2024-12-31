from infrastructure.envs.tabular_wrapper import EnvWrapper

import argparse
from datetime import datetime
import gymnasium as gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter

"""
    This is the interface we are going to use throughout the course.

    You will implement the RL algorithms as a subclasses of the common Trainer class.
    The only two required methods are `__init__` and `train`.

    The `Trainer::train` method should perform the training and return an object
    that implements the Policy interface. This is to enable automatic evaluation
    on our side, and to enable provided visualization utilities.
"""

class Policy:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int) -> int:
        raise NotImplementedError()

    # Raw output of the policy, this could later be logits/etc.
    # However, for this homework the output of raw(state) MUST be
    # the estimated value of the given state under the policy
    def raw(self, state : int) -> float:
        raise NotImplementedError()


class Trainer:
    # Stores the EnvWrapper object
    def __init__(self, env : EnvWrapper, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the number of iterations for VI, or total number of calls to env.step() for QL, SARSA, and MC
    def train(self, gamma : float, steps : int, **kwargs) -> Policy:
        raise NotImplementedError()

"""
    Note that an environment wrapper (for FrozenLake and CliffWalking) is
    prepared for you in infrastructure/envs/tabular_wrapper.py.

    The wrapper provides an interface for easily getting dynamics/reward
    information for both benchmarks. You should read through this interface
    before starting your implementation.

    The wrapper will also log average
    episode returns and lengths automatically, if logging is enabled.
    Run `tensorboard serve --logdir runs` to visualize the logs.

    The signatures of classes and methods that are provided (like VITrainer)
    should remain unchanged for evaluation purposes, however you are free to
    add your own methods, or your own training (hyper) parameters via kwargs. 
    Just make sure you provide default values for these parameters, i.e. via
    getting them by `kwargs.get(param_name, default_value)`, since during the
    evaluation we will only supply the positional parameters.

    One final note - the policies returned from SARSA and MC should `play()`
    epsilon-greedily with the value of epsilon that was supplied to the
    trainer. Since the visualization tools from the wrapper rely on a single 
    deterministic decision for each state, all your `play()` methods 
    should accept an optional parameter called `greedy`. 
    If `play(state, greedy=True)` 
    is called, the method should return the greedy action with respect to the 
    Q estimates for `state`.
"""

"""
    PROBLEM a): VALUE ITERATION
"""

class VITrainer(Trainer):

    def __init__(self, env, **kwargs):
        # `env` is saved as `self.env`
        super(VITrainer, self).__init__(env)

    def train(self, gamma, steps, **kwargs):
        # TODO - complete the Value Iteration Algorithm, 
        # and execute steps number of iterations

        # The states are numbers \in [0, ... nS-1], same with actions.
        nS = self.env.num_states()
        nA = self.env.num_actions()
        values = ...

        # recall that environment dynamics are available as full tensors:
        # w. `self.env.get_dynamics_tensor()`, or via `get_transition(s, a, s')`

        # Make sure you return an object extending the Policy interface, so
        # that you are able to render the policy and evaluate it.
        pass

"""
    PROBLEM b): Q-LEARNING
"""

class QLTrainer(Trainer):

    def __init__(self, env, **kwargs):
        super(QLTrainer, self).__init__(env)
        # feel free to add stuff here as well

    def train(self, gamma, steps, eps, lr, explore_starts=False, **kwargs):
        # TODO - complete the QLearning algorithm that uses the supplied
        # values of eps/lr (for the whole training). Use an epsilon-greedy exploration policy.

        # TODO: modify this call for exploring starts as well
        state, info = self.env.reset()
        done = False

        while not done:
            # TODO: action selection
            action = 0
            succ, rew, terminated, truncated, _ = self.env.step(action)


            # TODO: update values
            if terminated or truncated:
                done = True
        # TODO: remember to only perform `steps` samples from the training environment


"""
    PROBLEM c): SARSA
"""

class SARSATrainer(Trainer):

    def __init__(self, env, **kwargs):
        super(SARSATrainer, self).__init__(env)

    def train(self, gamma, steps, eps, lr, explore_starts=False, **kwargs):
        # TODO - complete the SARSA algorithm that uses the supplied values of
        # eps/lr and exploring starts.
        pass


"""
    PROBLEM d): EVERY VISIT MONTE CARLO CONTROL
"""

class MCTrainer(Trainer):
    def __init__(self, env, **kwargs):
        super(MCTrainer, self).__init__(env)

    def train(self, gamma, steps, eps, explore_starts=False, **kwargs):
        # TODO - Complete every visit MC-control, which uses an epsilon greedy
        # exploration policy
        pass

"""
    PROBLEM e) Evaluation
        
    As part of the last problem, you are expected to deliver visualizations
    of the learning curves of each algorithm on each environment.

    To achieve this, you can include a logging callback in the training loop of
    each of the algorithms that evaluates the current policy for a number of
    steps, logs statistics about the value of the inital state / avg
    episode return, etc. See the example below that makes use of Tensorboard. 

    Note that the training environments passed as input to each Trainer
    are limited in the number of samples you can take. You can use evaluation
    environments to generate the average episode rewards and logs instead.
    Evaluation environments are created by passing `evaluation=True` to the
    EnvWrapper constructor (see main).
"""

def logging_callback(evaluation_env, logger, policy, algorithm_name, step, gamma):

    """
        `logger` - SummaryWriter() instance created in main
        `step` - the timestep of the logging, x axis of the plot

        You can modify or delete this function, this is just a demonstration 
        of how you can achieve logging 
        your training data to a tensorboard file. You can modify a Dataframe,
        or log directly to a csv file instead.
    """

    initial_state, _ = evaluation_env.reset()

    # Some code to run a few episodes the policy on evaluation_env..
    avg_reward = 42.0

    data = {
        "avg_reward" : avg_reward,
        "std_reward" : 0.0,
        "value" : policy.raw(initial_state),
    }
    
    # add all data to returns/algorithm_name_*"
    logger.add_scalars(algorithm_name, data, step)


"""
    We will demonstrate the logging, as well as the rendering methods implemented
    in the wrapper using a dummy policy.
"""
class RandomPolicy(Policy):
    """
        A dummy policy that returns random actions and random values
    """
    def __init__(self, nA):
        self.nA = nA

    """
        remember the `greedy` flag to use the visualizations
        (even though it does nothing here)
    """
    def play(self, state, greedy=False):
        return np.random.randint(self.nA)

    def raw(self, state):
        return np.random.randint(42)


def render_random(env):
    """
        Plots heatmap of the state values and arrows corresponding to actions on `env`
    """
    env.reset(randomize=False)
    policy = RandomPolicy(env.num_actions())
    env.render_policy(policy, label= "RandomPolicy")



if __name__ == "__main__":

    """
        These parameters control the logging of episode rewards in the env
        wrapper. You may change these as you like.
    """ 
    logging_kwargs = {
                       "logging": True,
                       "log_dir" : "runs/", 
                       "log_title" : None,
                       "log_steps" : 500
                      }

    max_samples = 10000

    FrozenLake = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'),
                            max_samples = max_samples,
                            **logging_kwargs)

    LargeLake = EnvWrapper(gym.make('FrozenLake-v1', map_name='8x8'),
                            max_samples = max_samples,
                            **logging_kwargs)

    CliffWalking = EnvWrapper(gym.make('CliffWalking-v0'),
                            max_samples = max_samples,
                            **logging_kwargs)

    """
        Evaluation environments - these are not limited to `max_samples` and you
        can use these to evaluate your policy - both after and during the
        training.
    """
    FrozenLakeEval = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'),
                                evaluation=True,
                                logging=False)

    LargeLakeEval = EnvWrapper(gym.make('FrozenLake-v1', map_name='8x8'),
                               evaluation=True,
                               logging=False)

    CliffWalkingEval = EnvWrapper(gym.make('CliffWalking-v0'),
                                  evaluation=True,
                                  logging=False)


    """
        Logging example - randomly step through FrozenLake and log 
        every 500 steps
    """
    nA = FrozenLake.num_actions()
    policy = RandomPolicy(nA)

    current_timestamp = datetime.now()
    timestamp = current_timestamp.strftime('%Y-%m-%d-%H:%M:%S')
    logger = SummaryWriter(f"results/FrozenLake_Random" + timestamp) 

    i = 0
    gamma = 0.99

    obs, _ = FrozenLake.reset()
    while i < max_samples:
        done = False
        while not done and i < max_samples:
            obs, rew, term, trunc, _ = FrozenLake.step(policy.play(obs))
            i += 1

            if i % 500 == 0:
                # use the eval env to get episode rewards, etc.
                logging_callback(FrozenLakeEval, logger, policy, "Random", i,
                        gamma=gamma)

            if term or trunc:
                done = True
                FrozenLake.reset()

    """
        Rendering example - using env.render_policy() to get a value heatmap as
        well as the greedy actions w.r.t. the policy values.
    """
    render_random(FrozenLake)

    """ 
        You can also use the `render_mode="human"` argument for Gymnasium to
        see an animation of your agent's decisions.
    """
    AnimatedEnv = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'
                                                     , render_mode='human'),
                             max_samples = max_samples,
                             **logging_kwargs)
    AnimatedEnv.reset()

    for i in range(10):
        AnimatedEnv.step(np.random.randint(4))

