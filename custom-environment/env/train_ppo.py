

"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import torch
import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn 
from custom_environment import Spec_Ops_Env
    
def env_creator():
    env = Spec_Ops_Env()
    
    return env

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    return agent_id
    
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_gpus=1)

    env_name = "123"

    print("\n\n\nTORCH CUDA UNDA?:", torch.cuda.is_available(),'\n\n\n')

    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    #print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))


    env=env_creator()

    config = (
        PPOConfig()
        .environment(env="123", clip_actions=True)
        .rollouts(num_rollout_workers=7)#, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=env.possible_agents,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .debugging(log_level="ERROR")
        .framework(framework="tf")
        .resources(num_gpus=1)#int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    )


    # Get the user's home directory
    user_home = os.path.expanduser("/data2/Vaibhav/Vaibhav/RLH/custom-environment/env/loggs")
    local_dir = user_home

    tune.run(
        "PPO",
        name="PPO",
        #resources_per_trial={"cpu": 4, "gpu": 1}, 
        stop={"timesteps_total": 5000000},
        checkpoint_freq=50,
        local_dir=local_dir,
        config=config.to_dict(),
    )