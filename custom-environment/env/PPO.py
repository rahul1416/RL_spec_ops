import functools
import random
from copy import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import map
import custom_fov_algo
import gymnasium
# from gymnasium.spaces *
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import parallel_to_aec
from pettingzoo.test import parallel_api_test
from pettingzoo.utils import agent_selector, wrappers
from ray.rllib import MultiAgentEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn
from visualizer import Visualizer

#Default variables
MAP_SIZE = (15, 15)

def angle_from_agent(px, py, sx, sy): # (px,py) are the coordinates from which (sx,sy) angle is measured
    x = sx - px
    y = sy - py
    angle = math.atan2(y, x)
    if angle < 0:
        angle += 2 * math.pi
    return 360-180*(1/math.pi)*angle

def is_there(visible=None,corr_x=None,corr_y=None):
    for i in visible:
        if (corr_x,corr_y) in visible:
            return True
        else:
            return False

def env(**kwargs):
    env = Spec_Ops_Env(**kwargs)
    if env.continuous:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env



parallel_env = parallel_wrapper_fn(env)

class Spec_Ops_Env(ParallelEnv):
    metadata={
        "name":"custom_environment_v0",
    }

    def __init__(self, render_mode=None, config=None):
        '''
        we initialize the following:
        - the starting location of terrorist
        - the random initialization location of the dummy soldier
        - timestamp
        - possible agents

        these attributes should not be changed after initialization
        '''

        self.config = config or {}

        #Initializing
        self.possible_agents=["terrorist_"+str(i) for i in range(self.config.get('num_terr',1))]
        self.possible_agents.extend(["soldier_"+str(i) for i in range(self.config.get('num_sol',1))])

        self.observation_spaces=dict(zip(self.possible_agents, [MultiDiscrete([10]*6400)]*2))
        self.action_spaces=dict(zip(self.possible_agents, [Discrete(6)]*2))

        self.sol_visible=set() # currently visible coordinates
        self.s_visible=set() # all the previous memory
        self.terr_visible=set()  # currently visible coordinates
        self.t_visible=set() # all the previous memory
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(1,len(self.possible_agents)+1)))
        )
        # print(self.agent_name_mapping)

        self.timestamp=None
        self.max_timestamp = self.config.get('max_timestamp', 420)

        #initializing rendering screen
        self.render_mode = self.config.get('render_mode', 'ansi')    #Check clashing with render_mode variable
        self.map_size = self.config.get('map_size', MAP_SIZE)

        self.viz = Visualizer(grid=self.map_size, agents=self.possible_agents)

    def get_agent_ids(self):
        return self.possible_agents
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the starting point
        it needs to initialize the follownig attributes:
        - agents
        - timestamp
        - terrorist coordinates, angles
        - soldier coordinates, angles
        - observation
        - infos
        and must set up the environment so that render(), step(), and observe() can be called without an issue
        """
        np.random.seed(seed) if seed else print('No Seeding only CHAOS!!!!!')

        self.agents = self.possible_agents[:]
        self.timestamp=0

        self.state = {'map':map.read_map_file('Maps/map_1.txt')} 
        # in state terrorist is given 1 and soldier given as 2 when there are two agents
        for agent in self.agents:
            #VVIP NOTE: Handling for invalid inputs/Initialization required!
            self.state[agent] = {}
            x = -1
            y = -1
            while(self.state['map'][y][x]!=4 or x<0 or y<0):
                x = np.random.randint(0,self.map_size[1])
                y = np.random.randint(0,self.map_size[1])
            print("\n\n\nAGENT COORDINATES:", x,y,agent,'\n\n\n')
            self.state[agent]['x']=x #if self.state[agent]['x']<0 else self.state[agent]['x'] #Randomly place the terrorist on the grid, facing an arbitrary angle
            self.state[agent]['y']=y #if self.state[agent]['y']<0 else self.state[agent]['y']
            self.state[agent]['angle']=np.random.randint(0,359) #if self.state[agent]['angle']<0 else self.state[agent]['angle']
            self.state[agent]['fov']=self.config.get(agent,{'fov': 90})['fov']  #Should be <179 becoz of math!
            self.state[agent]['shoot_angle']=self.config.get(agent,{'shooting_angle': 15})['shooting_angle']
            self.state[agent]['hp'] = 100
            self.state['map'][self.state[agent]['y']][self.state[agent]['x']] = self.agent_name_mapping[agent] # updating the location oof soldier and terrorist in state map

            #Error Checking
            if(self.state[agent]['fov'] >= 180 or self.state[agent]['fov'] >= 180):
                print("invalid fov angle agent ki icchav, chusko bey")
                exit()
                
        # print(self.state)
        infos = dict({a: {} for a in self.agents})    #Just a dummy, we are not using it for nowz
        self.observations = self.update_observations()
        
        return self.observations, infos

    def step(self, actions):
        """
        takes in an action for the current agent (specified by the agent_selection)
        needs to update:
        - coordinates of terrorist
        - coordinates of soldier
        - rotation of the terrorist
        - rotation of the soldier
        - termination condition
        - rewards
        - timestamp
        - infos
        - truncations
        add any internl state  use by observe() or render()
        """

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}


        # execute actions to update the state and get updated action masks
        action_masks = self.move(actions)

        # Initialize termination conditions and rewards
        self.terminations = {a: False for a in self.agents}

        rewards = {a: 0 for a in self.agents}   # rewards for all agents are placed in the rewards dictionary to be returned

        # Calculate the rewards and punishments
        rewards = self.get_rewards(rewards)

        truncations = {a: False for a in self.agents}
        if self.timestamp > self.max_timestamp:
            rewards = {"soldier": 0, "terrorist": 0}    #IS THIS REQUIRED???
            truncations = {"soldier": True, "terrorist": True}
        self.timestamp += 1

        # Get observations for each agent
        self.observations = self.update_observations()
        # Get dummy infos (not used for now)
        infos = {a: {} for a in self.agents}

        if any(self.terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode != None and self.viz:
            self.render()

        return self.observations, rewards, self.terminations, truncations, infos

    def move(self, actions):
        action_masks = {}
        for agent in actions.keys():        #NOTE: ADD WALLS AND OTHER AGENT COLLISION SUPPORT
            action = actions[agent]

            #Making it's current position free
            self.state['map'][self.state[agent]['y']][self.state[agent]['x']] = 0
            #Move the agent along with failsafes
            if action == 0 and self.state[agent]['x'] > 0 and self.state['map'][self.state[agent]['y']][self.state[agent]['x']-1]==0:
                self.state[agent]['x'] -= 1 # left
            elif action == 1 and self.state[agent]['x'] < (self.map_size[0]-1) and self.state['map'][self.state[agent]['y']][self.state[agent]['x']+1]==0:
                self.state[agent]['x'] += 1 # right
            elif action == 2 and self.state[agent]['y'] > 0 and self.state['map'][self.state[agent]['y']-1][self.state[agent]['x']]==0:
                self.state[agent]['y'] -= 1 # top
            elif action == 3 and self.state[agent]['y'] < (self.map_size[1]-1) and self.state['map'][self.state[agent]['y']+1][self.state[agent]['x']]==0:
                self.state[agent]['y'] += 1 # bottom
            elif action == 4 :
                self.state[agent]['angle'] += 30 # rotate 30 degrees anti clockwise
                if self.state[agent]['angle']>360:
                    self.state[agent]['angle']=self.state[agent]['angle']-360
            elif action == 5 :
                self.state[agent]['angle'] -= 30 # rotate 30 degrees clockwise
                if self.state[agent]['angle']<0:
                    self.state[agent]['angle']=self.state[agent]['angle']+360

            #Marking the newly occupied position of agent in the state map
            self.state['map'][self.state[agent]['y']][self.state[agent]['x']] = self.agent_name_mapping[agent]
            #Generate Action masks
            action_mask = np.ones(6, dtype=np.int8) # Initialize action mask with all actions allowed
            if self.state[agent]['x'] == 0 or self.state['map'][self.state[agent]['y']][self.state[agent]['x']-1]!=0:
                action_mask[0] = 0 # moving left
            if self.state[agent]['x'] == self.map_size[1]-1 or self.state['map'][self.state[agent]['y']][self.state[agent]['x']+1]!=0:
                action_mask[1] = 0 #  moving right
            if self.state[agent]['y'] == 0 or self.state['map'][self.state[agent]['y']-1][self.state[agent]['x']]!=0:
                action_mask[2] = 0 #  moving up
            if self.state[agent]['y'] == self.map_size[0]-1 or self.state['map'][self.state[agent]['y']+1][self.state[agent]['x']]!=0:
                action_mask[3] = 0  # moving down

            action_masks[agent] = action_mask
        return action_masks


    def get_rewards(self, rewards=None):
        rewards = {a: 0 for a in self.agents}
        # return rewards
        reward_t={a: 0 for a in self.agents}
        reward_s={a: 0 for a in self.agents}
        #Calculate the rewards and punishments
        for agent in self.agents:

            if agent.split("_")[0]=="terrorist":

                for i in self.agents:
                    if i.split("_")[0]=="soldier":
                        angle_soldier = angle_from_agent(self.state[agent]['x'], self.state[agent]['y'], self.state[i]['x'], self.state[i]['y'])
                        # right most angles
                        ss1 = self.state[agent]['angle']-self.state[agent]['shoot_angle']/2 #self.terr_angle-self.shoot_angle/2
                        if(ss1<0) : ss1 = 360+ss1
                        tt1 = self.state[agent]['angle']-self.state[agent]['fov']/2 #self.terr_angle-self.terr_fov/2
                        if(tt1<0): tt1 = 360+tt1
                        # left most angles
                        ss2 = self.state[agent]['angle']+self.state[agent]['shoot_angle']/2 #self.terr_angle+self.shoot_angle/2
                        if(ss2>=360): ss2=ss2-360
                        tt2 = self.state[agent]['angle']+self.state[agent]['fov']/2 #self.terr_angle+self.terr_fov/2
                        if(tt2>=360): tt2=tt2-360
                        '''the scope of angle is 0<=angle<=359, there is no 360'''
                        if tt2>tt1 :
                            if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                reward_t={"soldier":-1, "terrorist":2}
                            elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                reward_t={"soldier":-3, "terrorist":3}
                                self.terminations = {a: True for a in self.agents}
                            elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # Soldier is between shooting angle and tt2
                                reward_t={"soldier":-1, "terrorist":2}
                            else:
                                reward_t={"soldier":2, "terrorist":-1} # Soldier and terrorist are not in the specified conditions
                        else:
                            if tt1>ss1:
                                if ((((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-1, "terrorist":2}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                    reward_t={"soldier":-3, "terrorist":3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # Soldier is between the shooting angle and leftmost field of view line
                                    reward_t={"soldier":-1, "terrorist":2}
                                else:
                                    reward_t={"soldier":2, "terrorist":-1}
                            elif ss1>ss2:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-1, "terrorist":2}
                                elif((((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-3, "terrorist":3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-1, "terrorist":2}
                                else:
                                    reward_t={"soldier":2, "terrorist":-1}
                            else:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-1, "terrorist":2}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                    reward_t={"soldier":-3, "terrorist":3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-1, "terrorist":2}
                                else:
                                    reward_t={"soldier":2, "terrorist":-1}
            else:
                for i in self.agents:
                    if i.split("_")[0]=="terrorist":

                        angle_soldier = angle_from_agent(self.state[agent]['x'], self.state[agent]['y'], self.state[i]['x'], self.state[i]['y']) #angle_from_agent(self.sol_x, self.sol_y, self.terr_x, self.terr_y)
                        # right most angles
                        ss1 = self.state[agent]['angle']-self.state[agent]['shoot_angle']/2 #self.sol_angle-self.shoot_angle/2
                        if(ss1<0) : ss1 = 360+ss1
                        tt1 = self.state[agent]['angle']-self.state[agent]['fov']/2 #self.sol_angle-self.terr_fov/2
                        if(tt1<0): tt1 = 360+tt1
                        # left most angles
                        ss2 = self.state[agent]['angle']+self.state[agent]['shoot_angle']/2 #self.sol_angle+self.shoot_angle/2
                        if(ss2>=360): ss2=ss2-360
                        tt2 = self.state[agent]['angle']+self.state[agent]['fov']/2 #self.sol_angle+self.terr_fov/2
                        if(tt2>=360): tt2=tt2-360
                        # print("soldier pov:",tt1,angle_soldier,tt2,self.sol_angle)

                        if tt2>tt1 :
                            if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in between right most shoot and fov line
                                reward_s={"soldier":2, "terrorist":-1}
                            elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in the shooting angle
                                reward_s={"soldier":3, "terrorist":-3}
                                self.terminations = {a: True for a in self.agents}
                            elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                reward_s={"soldier":2, "terrorist":-1}
                            else:
                                reward_s={"soldier":-1, "terrorist":2}
                        else:
                            if tt1>ss1:
                                if ((((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_s={"soldier":2, "terrorist":-1}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in the shooting angle
                                    reward_s={"soldier":3, "terrorist":-3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":2, "terrorist":-1}
                                else:
                                    reward_s={"soldier":-1, "terrorist":2}
                            elif ss1>ss2:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in between right most shoot and fov line
                                    reward_s={"soldier":2, "terrorist":-1}
                                elif((((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":3, "terrorist":-3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":2, "terrorist":-1}
                                else:
                                    reward_s={"soldier":-1, "terrorist":2}
                            else:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":2, "terrorist":-1}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":3, "terrorist":-3}
                                    self.terminations = {a: True for a in self.agents}
                                elif((((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":2, "terrorist":-1}
                                else:
                                    reward_s={"soldier":-1, "terrorist":2}
        for i in (rewards.keys()):
            rewards[i]=reward_s[i.split("_")[0]]+reward_t[i.split("_")[0]]
        self.sol_visible.clear()
        self.terr_visible.clear()
        return rewards 

    def update_observations(self):
        # New code giving entire state to both agents
        obs={}
        for agent in self.agents:
            if agent.split("_")[0]=="soldier":
                obs_map=self.state['map'].copy()
                obs[agent]=obs_map.flatten()
            else:
                obs_map=self.state['map'].copy()
                obs[agent]=obs_map.flatten()
        return obs
    


        # -1 for wall, 1 for terrorist, 2 for soldier, 0 for empty space, 3 for unknown region
        obs={}
        def is_blocking(x, y):
            if((x<0) or (x>=80) or (y<0) or (y>=80)):
                return True
            elif((self.state['map'][y][x] != 0)):
                return True
            return False
        for agent in self.agents:
            if agent.split("_")[0]=="soldier":
                def reveal(x, y):
                    if x>=0 and y>=0 and x<=self.map_size[1] and y<self.map_size[0]:
                        self.sol_visible.add((x, y))
                        self.s_visible.add((x,y))
                obs_map=self.state['map'].copy()
                custom_fov_algo.compute_fov((self.state[agent]['x'],self.state[agent]['y']), self.state[agent]['angle'], self.state[agent]['fov'], is_blocking, reveal)
                for i in range(obs_map.shape[0]):
                    for j in range(obs_map.shape[1]):
                        if((j,i) in self.s_visible):
                            pass
                        else:
                            obs_map[i][j]=3
                obs[agent]=obs_map.flatten()
            else:
                def reveal(x, y):
                    if x>=0 and y>=0 and x<=self.map_size[1] and y<self.map_size[0]:
                        self.terr_visible.add((x, y))
                        self.t_visible.add((x,y))
                obs_map=self.state['map'].copy()
                custom_fov_algo.compute_fov((self.state[agent]['x'],self.state[agent]['y']), self.state[agent]['angle'], self.state[agent]['fov'], is_blocking, reveal)
                for i in range(obs_map.shape[0]):
                    for j in range(obs_map.shape[1]):
                        if((j,i) in self.t_visible):
                            pass
                        else:
                            obs_map[i][j]=3
                obs[agent]=obs_map.flatten()
        return obs

    def render(self):
        """Renders the environment."""
        self.viz.update(self.state, self.agents)
        time.sleep(0.1)

    def close(self):
        """
        CLose releases the pygame graphical display when env is no longer being used.
        """
        self.viz.quit()
        pass

from pettingzoo.test import api_test

if __name__ == '__main__':
    env=Spec_Ops_Env()
    parallel_api_test(env, num_cycles=1000)

    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
    env.close()
