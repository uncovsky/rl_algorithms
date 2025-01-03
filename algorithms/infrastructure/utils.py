
import gymnasium as gym
import numpy as np
import random
import torch

# Tensors are by default stored on cpu
device = torch.device('cpu')

# Lists available GPUs on this machine, along with their indices
# passing the respective index to init_device sets device for all tensors to be
# the given gpu

def list_devices():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print("Available gpus along with their device indices:")
        for i in range(num_gpus):
            print(f"GPU num {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA unavailable")

# Initializes device to gpu automatically if available
def init_device(gpu_idx=0):
    global device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_idx}')
        name = torch.cuda.get_device_name(gpu_idx)
        print(f"Setting device to GPU {gpu_idx}: {name}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU.")

def use_cpu():
    global device
    device = torch.device('cpu')


# detach from computation graph, move to cpu, convert to np
def to_numpy(torch_tensor):
    return torch_tensor.detach().cpu().numpy()

def to_torch(obj):
    return torch.from_numpy(obj).to(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


"""
    Helper function to get dimensions of state/action spaces of gym environments.
"""
def get_env_dimensions(env):

    def get_space_dimensions(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        else:
            raise TypeError(f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

    state_dim = get_space_dimensions(env.observation_space)
    num_actions = get_space_dimensions(env.action_space)

    return state_dim, num_actions
