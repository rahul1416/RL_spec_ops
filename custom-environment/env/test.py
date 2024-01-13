import os

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from custom_environment import Spec_Ops_Env
    
def env_creator():
    env = Spec_Ops_Env()    
    return env

def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    return agent_id

ray.init(ignore_reinit_error=True, num_gpus=1)

env_name = "123"

#print("\n\n\nNEE YABBA TORCH CUDA UNDA?:", torch.cuda.is_available(),'\n\n\n')
register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))

env=env_creator()

config = (
        PPOConfig()
        .environment(env="123", clip_actions=True)
        .rollouts(num_rollout_workers=1)
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
        .resources(num_gpus=1)
    )

from ray.rllib.algorithms.algorithm import Algorithm

def new_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    print(agent_id)
    return  agent_id


algo_w_2_policies = Algorithm.from_checkpoint(
    checkpoint='/data2/Vaibhav/Vaibhav/RLH/custom-environment/env/loggs/DQN/DQN_123_78a9e_00000_0_2023-11-21_20-57-24/checkpoint_000086',
    policy_ids=["terrorist_0", "soldier_0"],
    policy_mapping_fn=policy_map_fn, 
)

obs=env.reset()

import time
env.reset()
while True:
    if(type(obs) == type(())):
        terr_a = algo_w_2_policies.compute_single_action(obs[0]['terrorist_0'], policy_id="terrorist_0")
        sol_a = algo_w_2_policies.compute_single_action(obs[0]['soldier_0'], policy_id="soldier_0")
    else:
        terr_a = algo_w_2_policies.compute_single_action(obs['terrorist_0'], policy_id="terrorist_0")
        sol_a = algo_w_2_policies.compute_single_action(obs['soldier_0'], policy_id="soldier_0")
    obs, rewards, terminations, truncations, infos = env.step({"terrorist_0": terr_a, "soldier_0": sol_a})
    env.render()
    print(obs)
    print(terminations, truncations)
    if any(terminations.values()) or all(truncations.values()):
        break
ray.shutdown()
exit()