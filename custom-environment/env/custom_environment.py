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

        self.observation_spaces=dict(zip(self.possible_agents, [MultiDiscrete([4]*8)]*2))
        self.action_spaces=dict(zip(self.possible_agents, [Discrete(6)]*2))

        self.sol_visible=set() # currently visible coordinates
        self.s_visible=set() # all the previous memory
        self.terr_visible=set()  # currently visible coordinates
        self.t_visible=set() # all the previous memory
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(1,len(self.possible_agents)+1)))
        )
        # print(self.agent_name_mapping)
        self.sol_hp=2
        self.terr_hp=2
        self.timestamp=None
        self.max_timestamp = self.config.get('max_timestamp', 469)

        #initializing rendering screen
        self.render_mode = self.config.get('render_mode', 'ansi')    #Check clashing with render_mode variable
        self.map_size = self.config.get('map_size', MAP_SIZE)

        self.vizz = True
        if(self.vizz):
            self.viz = Visualizer(grid=self.map_size, agents=self.possible_agents)

        # Reward System $$
        self.terr_visited_lat_y = [1]*self.map_size[0]    #Terrorist lat long policy
        self.terr_visited_long_x = [1]*self.map_size[1]

        self.sol_visited_map = [[0]*self.map_size[0]]*self.map_size[1]    #Soldier exploration reward keliye

        self.terr_last_known_pos = {'x':14, 'y':0}  #For rewarding getting closer to terrorist
        # Reward system design done

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
        if(seed):
            np.random.seed(seed) 

        self.agents = self.possible_agents[:]
        self.timestamp=0

        self.state = {'map':map.read_map_file('Maps/map_1.txt')} #{"map": np.zeros((self.config.get('map_size', MAP_SIZE)))}
        # in state terrorist is given 1 and soldier given as 2 when there are two agents
        for agent in self.agents:
            #VVIP NOTE: Handling for invalid inputs/Initialization required!
            self.state[agent] = {}
            x = -1
            y = -1
            while(self.state['map'][y][x]!=0 or x<0 or y<0):
                x = np.random.randint(0,self.map_size[1])
                y = np.random.randint(0,self.map_size[1])
            self.state[agent]['x']=x  #Randomly place the terrorist on the grid, facing an arbitrary angle
            self.state[agent]['y']=y 
            self.state[agent]['angle']=np.random.randint(0,359) 
            self.state[agent]['fov']=self.config.get(agent,{'fov': 90})['fov']  #Should be <179 becoz of math!
            self.state[agent]['shoot_angle']=self.config.get(agent,{'shooting_angle': 15})['shooting_angle']
            if agent=="soldier_0":
                self.state[agent]['hp'] = self.sol_hp
            else:
                self.state[agent]['hp'] = self.terr_hp
            self.state['map'][self.state[agent]['y']][self.state[agent]['x']] = self.agent_name_mapping[agent] # updating the location oof soldier and terrorist in state map
            if(self.state[agent]['fov'] >= 180 or self.state[agent]['fov'] >= 180):
                exit()
        self.terr_last_known_pos = {'x':self.state['terrorist_0']['x'],'y':self.state['terrorist_0']['y']}
        infos = dict({a: {} for a in self.agents})    #Just a dummy, we are not using it for now
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
        collision_rewards = self.move(actions)

        # Initialize termination conditions and rewards
        self.terminations = {a: False for a in self.agents}
        # Get observations for each agent
        self.observations = self.update_observations()

        rewards = {a: 0 for a in self.agents}   # rewards for all agents are placed in the rewards dictionary to be returned


        # Get observations for each agent



        # Calculate the rewards and punishments
        rewards = self.get_rewards(rewards, collision_rewards)
        if self.sol_hp==0 or self.terr_hp==0:
            self.terminations={a: True for a in self.agents}
        truncations = {a: False for a in self.agents}
        if self.timestamp > self.max_timestamp:
            print('done 469 steps hence ')
            rewards = {"soldier": 0, "terrorist": 0}    #IS THIS REQUIRED???
            truncations = {"soldier": True, "terrorist": True}
        self.timestamp += 1

        # Get dummy infos (not used for now)
        infos = {a: {} for a in self.agents}

        if any(self.terminations.values()) or all(truncations.values()):
            self.agents = []

        if self.render_mode != None and self.vizz:
            self.render()

        return self.observations, rewards, self.terminations, truncations, infos

    def move(self, actions):
        #action_masks = {}
        collision_rewards = {}
        for agent in actions.keys():        #NOTE: ADD WALLS AND OTHER AGENT COLLISION SUPPORT - DONE
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

            # Calculating collision punishments:
            reward = 0
            if action == 0 and (self.state[agent]['x'] <= 0 or self.state['map'][self.state[agent]['y']][self.state[agent]['x']-1]!=0):
                reward=-1   # Left
            elif action == 1 and (self.state[agent]['x'] >= (self.map_size[0]-1) or self.state['map'][self.state[agent]['y']][self.state[agent]['x']+1]!=0):
                reward=-1 # right
            elif action == 2 and (self.state[agent]['y'] <= 0 or self.state['map'][self.state[agent]['y']-1][self.state[agent]['x']]!=0):
                reward=-1 # top
            elif action == 3 and (self.state[agent]['y'] >= (self.map_size[1]-1) or self.state['map'][self.state[agent]['y']+1][self.state[agent]['x']]!=0):
                reward=-1 # bottom
            collision_rewards[agent] = reward

            # Soldier visited grid counter
            if agent=="soldier_0":
                self.sol_visited_map[self.state[agent]['y']][self.state[agent]['x']]+=1

            #Marking the newly occupied position of agent in the state map
            self.state['map'][self.state[agent]['y']][self.state[agent]['x']] = self.agent_name_mapping[agent]
        return collision_rewards


    def get_rewards(self, rewards=None, collision_rewards=None):
        rewards = {a: 0 for a in self.agents}
        # return rewards
        reward_t={a.split('_')[0]: 0 for a in self.agents}
        reward_s={a.split('_')[0]: 0 for a in self.agents}
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
                        # print("terrorist pov:",tt1,angle_soldier,tt2,self.terr_angle)
                        '''the scope of angle is 0<=angle<=359, there is no 360'''
                        if tt2>tt1 :
                            if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                reward_t={"soldier":-5, "terrorist":10}
                            elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                reward_t={"soldier":-15, "terrorist":20}
                                self.sol_hp-=1
                            elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                reward_t={"soldier":-5, "terrorist":10}
                        else:
                            if tt1>ss1:
                                if ((((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-5, "terrorist":10}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                    reward_t={"soldier":-15, "terrorist":20}
                                    self.sol_hp-=1
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-5, "terrorist":10}
                            elif ss1>ss2:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-5, "terrorist":10}
                                elif((((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-15, "terrorist":20}
                                    self.sol_hp-=1
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-5, "terrorist":10}
                            else:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_t={"soldier":-5, "terrorist":10}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in the shooting angle
                                    reward_t={"soldier":-15, "terrorist":20}
                                    self.sol_hp-=1
                                elif((((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))) and is_there(self.terr_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_t={"soldier":-5, "terrorist":10}
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

                        if tt2>tt1 :
                            if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in between right most shoot and fov line
                                reward_s={"soldier":10, "terrorist":-5}
                            elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in the shooting angle
                                reward_s={"soldier":20, "terrorist":-15}
                                self.terr_hp-=1                         
                            elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                reward_s={"soldier":10, "terrorist":-5}
                        else:
                            if tt1>ss1:
                                if ((((angle_soldier>=tt1) and (angle_soldier>ss1)) or ((angle_soldier<tt1) and (angle_soldier<ss1))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # soldier in between right most shoot and fov line
                                    reward_s={"soldier":10, "terrorist":-5}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in the shooting angle
                                    reward_s={"soldier":20, "terrorist":-15}
                                    self.terr_hp-=1                               
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":10, "terrorist":-5}
                            elif ss1>ss2:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])): # terrorist in between right most shoot and fov line
                                    reward_s={"soldier":10, "terrorist":-5}
                                elif((((angle_soldier>=ss1) and (angle_soldier>ss2)) or ((angle_soldier<ss1) and (angle_soldier<=ss2))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":20, "terrorist":-15}
                                    self.terr_hp-=1
                                elif((angle_soldier>ss2) and (angle_soldier<=tt2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":10, "terrorist":-5}
                            else:
                                if ((angle_soldier>=tt1) and (angle_soldier<ss1) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":10, "terrorist":-5}
                                elif((angle_soldier>=ss1) and (angle_soldier<=ss2) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":20, "terrorist":-15}
                                    self.terr_hp-=1
                                elif((((angle_soldier>tt2) and (angle_soldier>ss2)) or ((angle_soldier<=tt2) and (angle_soldier<ss2))) and is_there(self.sol_visible,self.state[i]['x'],self.state[i]['y'])):
                                    reward_s={"soldier":10, "terrorist":-5}
        for i in (rewards.keys()):
            rewards[i]=reward_s[i.split("_")[0]]+reward_t[i.split("_")[0]]
        
        sol_explore_reward=(1.0/((self.sol_visited_map[self.state['soldier_0']['y']][self.state['soldier_0']['x']]+1)*22))
        euc_dist=abs(self.state['soldier_0']['x']-self.terr_last_known_pos['x'])+abs(self.state['soldier_0']['y']-self.terr_last_known_pos['y'])
        
        terr_closeness_reward=15/(euc_dist+1)
        rewards['soldier_0']+=sol_explore_reward+terr_closeness_reward

        rewards['terrorist_0']+=self.terr_visited_lat_y[self.state['terrorist_0']['y']]
        if(self.terr_visited_lat_y[self.state['terrorist_0']['y']] == 1):
            self.terr_visited_lat_y[self.state['terrorist_0']['y']] = 0
        rewards['terrorist_0']+=self.terr_visited_long_x[self.state['terrorist_0']['x']]
        if(self.terr_visited_long_x[self.state['terrorist_0']['x']] == 1):
            self.terr_visited_long_x[self.state['terrorist_0']['x']] = 0
        for agent in self.agents:
            rewards[agent] += collision_rewards[agent]

        self.sol_visible.clear()
        self.terr_visible.clear()
        print('hp sol, terr:', self.sol_hp, self.terr_hp)
        return rewards 

    def update_observations(self):
        obs={}
        def is_blocking(x, y):
            if((x<0) or (x>=self.map_size[0]) or (y<0) or (y>=self.map_size[1])):
                return True
            elif((self.state['map'][y][x] != 0)):
                return True
            return False
        check_these=[-2,-1,1,2,-2,-1,1,2]
        
        for agent in self.agents:
            if agent.split("_")[0]=="soldier":
                def reveal(x, y):
                    if x>=0 and y>=0 and x<=self.map_size[1] and y<self.map_size[0]:
                        self.sol_visible.add((x, y))
                obs_map=self.state['map'].copy()
                custom_fov_algo.compute_fov((self.state[agent]['x'],self.state[agent]['y']), self.state[agent]['angle'], self.state[agent]['fov'], is_blocking, reveal)
                sol_obs=[]
                for i in range(len(check_these)):
                    if i<4:
                        if self.state[agent]['x']+check_these[i]<0:
                            sol_obs.append(3)
                        elif self.state[agent]['x']+check_these[i]>(self.map_size[1]-1):
                            sol_obs.append(3)
                        else:
                            sol_obs.append(obs_map[self.state[agent]['y']][self.state[agent]['x']])
                    else:
                        if self.state[agent]['y']+check_these[i]<0:
                            sol_obs.append(3)
                        elif self.state[agent]['y']+check_these[i]>(self.map_size[0]-1):
                            sol_obs.append(3)
                        else:
                            sol_obs.append(obs_map[self.state[agent]['y']][self.state[agent]['x']])

                obs[agent]=np.array(sol_obs)
            else:

                def reveal(x, y):
                    if x>=0 and y>=0 and x<=self.map_size[1] and y<self.map_size[0]:
                        self.terr_visible.add((x, y))

                obs_map=self.state['map'].copy()
                custom_fov_algo.compute_fov((self.state[agent]['x'],self.state[agent]['y']), self.state[agent]['angle'], self.state[agent]['fov'], is_blocking, reveal)
                terr_obs=[]
                for i in range(len(check_these)):
                    if i<4:
                        if self.state[agent]['x']+check_these[i]<0:
                            terr_obs.append(3)
                        elif self.state[agent]['x']+check_these[i]>(self.map_size[1]-1):
                            terr_obs.append(3)
                        else:
                            terr_obs.append(obs_map[self.state[agent]['y']][self.state[agent]['x']])
                    else:
                        if self.state[agent]['y']+check_these[i]<0:
                            terr_obs.append(3)
                        elif self.state[agent]['y']+check_these[i]>(self.map_size[0]-1):
                            terr_obs.append(3)
                        else:
                            terr_obs.append(obs_map[self.state[agent]['y']][self.state[agent]['x']])
          
                obs[agent]=np.array(terr_obs)
        if ((self.state['terrorist_0']['x'],self.state['terrorist_0']['y']) in self.sol_visible):
            self.terr_last_known_pos['x']=self.state['terrorist_0']['x']
            self.terr_last_known_pos['y']=self.state['terrorist_0']['y']
        return obs

    def render(self):
        if(self.vizz == False):
            return
        """Renders the environment."""

        self.viz.update(self.state, self.agents, self.sol_hp, self.terr_hp)
        time.sleep(0.5)

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
