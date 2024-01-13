"""
* In a loop call the environmeht.
* In reality, the environment will be called by RLlib, but doing it manually allows us to test this
  part and see what is going on.
"""
from typing import Dict
import numpy as np
import time
import random

from ray.rllib.env import EnvContext
from custom_environment import Spec_Ops_Env
# from your_constants import NUM_CHICKENS, NUM_DIRECTIONS
# from your_rllib_environment import YourEnvironment


def is_all_done(done: Dict) -> bool:
    for key, val in done.items():
        if not val:
            return False
    return True


env_config = {}
config = EnvContext(env_config, worker_index=1)

env = Spec_Ops_Env(config)
env.reset()

action_dict = {
    'terrorist_0':np.random.randint(6),
    'soldier_0':np.random.randint(6)
}

obs, rew, done, info = env.step(action_dict)
env.render()

while not is_all_done(done):
    action_dict = {}
    assert 'terrorist_0' in obs or 'soldier_0' in obs
    if 'terrorist_0' in obs and not done['terrorist_0']:
        action_dict['terrorist_0'] = random.choice(range(6))
    if 'soldier_0' in obs and not done['soldier_0']:
        action_dict['soldier_0'] = random.choice(range(6))
    obs, rew, done, info = env.step(action_dict)
    print("Reward: ", rew)
    time.sleep(.1)
    env.render()


time.sleep(50)