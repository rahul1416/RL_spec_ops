import pygame
import math
from fractions import Fraction
import sys
import map
import os

# Define the colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255,255,255)
YELLOW = (255, 255, 0)

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Visualizer():
    def __init__(self,grid=(10,10), caption="Spec Ops Visualization", screen_dim=(800, 800), agents=None):
        # Initialize Pygame
        pygame.init()

        # Create the screen
        self.grid = grid
        self.screen_dim = screen_dim
        self.screen = pygame.display.set_mode(self.screen_dim)

        # Set the caption
        self.caption = caption
        pygame.display.set_caption(self.caption)

        #Initialize agents
        self.agents = agents
        if(self.agents == None):
            print("VISUALIZER ERROR: No agents given in Visualizer!!")
            exit()

    def update(self,state=None, reward={"soldier":0,"terrorist":0}):
        # Check if a state is provided
        if(state == None):
            print("VISUALIZER ERROR: No state given to render!!")
            exit()

        # Handle pygame events
        for event in pygame.event.get():
            # Quit the game if the window is closed
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Handle player input for movement and actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    state['soldier_player']['x'] -= 1
                if event.key == pygame.K_RIGHT:
                    state['soldier_player']['x'] += 1
                if event.key == pygame.K_UP:
                    state['soldier_player']['y'] -= 1
                if event.key == pygame.K_DOWN:
                    state['soldier_player']['y'] += 1
                if event.key == pygame.K_a: # Rotate player clockwise
                    state['soldier_player']['angle'] += 10
                    if(state['soldier_player']['angle']>360):
                        state['soldier_player']['angle'] -= 360
                if event.key == pygame.K_d:
                    state['soldier_player']['angle'] -= 10 # Rotate player counterclockwise
                    if(state['soldier_player']['angle']<0):
                        state['soldier_player']['angle'] += 360
                if event.key == pygame.K_q: # Fine-tune clockwise rotation
                    state['soldier_player']['angle'] += 5
                    if(state['soldier_player']['angle']>360):
                        state['soldier_player']['angle'] -= 360
                if event.key == pygame.K_e: # Fine-tune counterclockwise rotation   
                    state['soldier_player']['angle'] -= 5
                    if(state['soldier_player']['angle']<0):
                        state['soldier_player']['angle'] += 5

                # Call custom FOV algorithm
                custom_fov_algo.compute_fov((state['soldier_player']['x'],state['soldier_player']['y']), state['soldier_player']['angle'], state['soldier_player']['fov'], is_blocking, reveal)


        #Draw white background
        self.screen.fill(WHITE)

        #Draw the grid
        w = int(self.screen_dim[0]/self.grid[0])
        for i in range(0, self.screen_dim[0], int(self.screen_dim[0]/self.grid[0])):
            pygame.draw.line(self.screen, BLACK, (i, 0), (i, self.screen_dim[0]))
            pygame.draw.line(self.screen, BLACK, (0, i), (self.screen_dim[0], i))

        # Draw walls and visible areas
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                if (i,j) in is_visible:
                    pass
                    pygame.draw.rect(self.screen,YELLOW,(w*i,w*j,w,w))
                if state['map'][j][i] == -1 and (i,j) in is_visible:
                    pygame.draw.rect(self.screen,BLACK,(w*i,w*j,w,w))

        # Draw agents (soldiers and terrorists)
        for agent in self.agents:
            agent_name = agent
            agent = AttributeDict(state[agent])
            if('soldier' in agent_name):
                # Draw soldier agent
                a = 20
                pa = agent.angle
                pa = math.pi*pa/180
                px, py = (2*agent.x+1)*(self.screen_dim[0]/self.grid[0])/2, (2*agent.y+1)*(self.screen_dim[0]/self.grid[0])/2
                fx, fy = ((px+2*a*math.sin(pa),py+2*a*math.cos(pa)))
                signx = (-1,1)[(pa>=270 or pa<=90)]
                signy = (-1,1)[(pa>=180)]
                l = 1000
                fov = (agent.fov)*(math.pi/180)
                shoot_angle=(agent.shoot_angle)*(math.pi/180)
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa+fov/2), py+signy*l*math.sin(pa+fov/2)))
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa-fov/2), py+signy*l*math.sin(pa-fov/2)))


                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(math.pi/4), py+signy*l*math.sin(math.pi/4)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(3*math.pi/4), py+signy*l*math.sin(3*math.pi/4)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(5*math.pi/4), py+signy*l*math.sin(5*math.pi/4)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(7*math.pi/4), py+signy*l*math.sin(7*math.pi/4)))
                pa += math.pi/2
                pygame.draw.polygon(self.screen, (0,0,255), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))

                pygame.draw.rect(self.screen,BLACK,(w*agent.x,w*agent.y,w,w))
                
            elif('terrorist' in agent_name):
                # Draw terrorist agent
                a = 20
                pa = agent.angle
                pa = math.pi*pa/180
                px, py = (2*agent.x+1)*(self.screen_dim[0]/self.grid[0])/2, (2*agent.y+1)*(self.screen_dim[0]/self.grid[0])/2
                fx, fy = ((px+2*a*math.sin(pa),py+2*a*math.cos(pa)))
                signx = (-1,1)[(pa>270 or pa<90)]
                signy = (-1,1)[(pa>180)]
                l = 1000
                fov = (agent.fov)*(math.pi/180)
                shoot_angle=(agent.shoot_angle)*(math.pi/180)
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa+fov/2), py+signy*l*math.sin(pa+fov/2)))
                pygame.draw.line(self.screen,BLACK,(px,py),(px+signx*l*math.cos(pa-fov/2), py+signy*l*math.sin(pa-fov/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa+shoot_angle/2), py+signy*l*math.sin(pa+shoot_angle/2)))
                pygame.draw.line(self.screen,RED,(px,py),(px+signx*l*math.cos(pa-shoot_angle/2), py+signy*l*math.sin(pa-shoot_angle/2)))
                pa += math.pi/2
                pygame.draw.polygon(self.screen, (255,0,0), ((px+a*math.sin(pa),py+a*math.cos(pa)), (px+a*math.sin(math.pi/3-pa), py-a*math.cos(math.pi/3-pa)), (px-a*math.sin(2*math.pi/3-pa), py+a*math.cos(2*math.pi/3-pa))))

        # Update the screen
        pygame.display.flip()

    def quit(self):
        # Quit the pygame module
        pygame.quit()

import custom_fov_algo
state = {'map':map.read_map_file('Maps/map_0.txt')}
state['soldier_player'] = {
    "x":40,
    "y":20,
    "angle":125,
    "shoot_angle":90,
    "fov":90
}

state['terrorist_player'] = {
    "x":40,
    "y":40,
    "angle":125,
    "shoot_angle":120,
    "fov":120
}

def is_blocking(x, y):
    if((x<0) or (x>=80) or (y<0) or (y>=80)):
      return True
    elif((state['map'][y][x] != 0)):
      return True
    return False


is_visible = set()

def reveal(x, y):
    is_visible.add((x, y))


if __name__ == '__main__':
    #CLI Rendering
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in state['map']:
        for j in i:
            if(j==0):
                print(j, end='')
            else:
                print(j, end='') 
        print()
    print('------------------------------------\n\n\n')
    agents = ['soldier_player', 'terrorist_player']
    state['map'][state['terrorist_player']['y']][state['terrorist_player']['x']] = -1  # represent that terrorist is "-1"
    viz = Visualizer(grid=(80,80), agents=agents)
    custom_fov_algo.compute_fov((state['soldier_player']['x'],state['soldier_player']['y']), state['soldier_player']['angle'], state['soldier_player']['fov'], is_blocking, reveal)
    while(True):
      viz.update(state, agents)

