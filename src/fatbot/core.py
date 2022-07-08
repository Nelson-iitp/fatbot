#==============================================================
import numpy as np
from math import ceil, inf, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#import pandas as pd
import gym
from .common import get_nspace, get_angle, REMAP
#==============================================================

class World(gym.Env):
    
    # Data-types
    STATE_DTYPE =    np.float32
    ACTION_DTYPE =   np.float32
    REWARD_DTYPE =   np.float32

    def __init__(self, name, world_info, bot_info, initial_states, horizon=0, enable_imaging=False, seed=None) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)  
        self._max_episode_steps = (horizon if horizon>0 else inf)
        self.enable_imaging = enable_imaging
        self.name = name
        self.world_info =world_info
        self.bot_info = bot_info
        self.initial_states = initial_states
        self.initial_states_count = len(initial_states)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.build() # call default build
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    """ Section: Build """

    def build(self):
        # --> call in order
        self.build_params()
        self.build_observation_space()
        self.build_action_space()
        self.build_vectors()
        self.set_reward_signal()
        self.build_reward_signal()
        self.build_render_modes()
        self.build_meta() #<----- this can be called after __init___ to override defaults

    def build_params(self):
        
        # TIP: dont use setattr here (auto complete doesnt pick those)

        # read variables (params) 
        self.N_BOTS = int(self.world_info['N_BOTS'])
        if self.N_BOTS<1:
            raise Exception('Require at least 1 bot!')
        self.X_RANGE = float(self.world_info['X_RANGE'])
        self.Y_RANGE =  float(self.world_info['Y_RANGE'])
        self.SCAN_RADIUS = float(self.world_info['SCAN_RADIUS'])
        self.BOT_RADIUS = float(self.world_info['BOT_RADIUS'])
        self.SAFE_CENTER_DISTANCE = float(self.world_info['SAFE_DISTANCE']) + self.BOT_RADIUS * 2
        self.SPEED_LIMIT = float(self.world_info['SPEED_LIMIT'])
        self.DELTA_SPEED = float(self.world_info['DELTA_SPEED'])
        self.delta_action_mode = (self.DELTA_SPEED!=0.0)
        self.TARGET_RADIUS = float(self.world_info['TARGET_RADIUS'])
        # NOTE: # resolution for distance sensor images 
        # range-finder or distance sensor covers a circumference of 2*pi*SCAN_RADIUS, 
        #       so total pixels in image = circumference*SENSOR_RESOULTION
        self.SENSOR_RESOULTION = float(self.world_info['SENSOR_RESOULTION']) # pixels per unit distance  
        self.SENSOR_UNIT_ARC = self.SENSOR_RESOULTION * self.SCAN_RADIUS
        self.SENSOR_IMAGE_SIZE = int(ceil(2 * np.pi * self.SENSOR_UNIT_ARC)) # total pixels in sensor data input


        # for rendering
        self.MAX_SPEED = sqrt(2) * self.SPEED_LIMIT # note - this is magnitude of MAX velocity vector
        self.colors = self.bot_info['COLOR']
        self.markers = self.bot_info['MARKER']
        self.names = self.bot_info['NAME']
        self.img_aspect = self.SENSOR_IMAGE_SIZE/self.SENSOR_RESOULTION # for rendering
        self.arcDivs = 4 + 1  # no of divisions on sensor image
        self.arcTicks = np.array([ (int( i * self.SENSOR_UNIT_ARC), round(i*180/pi, 2)) \
                                    for i in np.linspace(0, 2*np.pi, self.arcDivs) ])
        return

    def build_observation_space(self):
        # now define observation_space ----------------------------------------------------------------------------
        position_space_info = {  # Position/Direction x,y,d   
                        'dim': 2,
                        'low': (-self.X_RANGE, -self.Y_RANGE), 
                        'high':( self.X_RANGE, self.Y_RANGE), 
                    }
        velocity_space_info = { # velocities dx, dy
                        'dim': 2,
                        'low': (-self.SPEED_LIMIT, -self.SPEED_LIMIT), 
                        'high':( self.SPEED_LIMIT,  self.SPEED_LIMIT), 
                    }
        neighbour_space_info = { # total_neighbours,  no of visible neighbours 
                        'dim': 3,  
                        'low': (0,           0,           0          ), 
                        'high':(self.N_BOTS, self.N_BOTS, self.N_BOTS), 
                    }
        self.observation_space = get_nspace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
                    shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'], ),
                    low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low']  ),
                    high=   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] ))
        
        self.o_dim =7
        self.base_observation = np.zeros(self.observation_space.shape, self.observation_space.dtype)
        self.observation = self.base_observation.reshape((self.N_BOTS, self.o_dim))
        self.initial_observation = np.zeros_like(self.observation)

        # define observation views
        self.xy =           self.observation [:, 0:2] # x,y
        self.x =            self.observation [:, 0:1] # x
        self.y =            self.observation [:, 1:2] # y

        self.dxy =           self.observation [:, 2:4] # dx,dy
        self.dx =            self.observation [:, 2:3] # dx
        self.dy =            self.observation [:, 3:4] # dy

        self.alln =            self.observation[:, 4:5]
        self.occn =            self.observation[:, 5:6]
        self.a =               self.observation[:, 6:7] # alive
        return

    def build_action_space(self):
        # now define action_space -------------------------------------------------------------------------------
        # action space can be of following types - 

        self.a_dim = 2
        if self.delta_action_mode:
            #   ~ change linear and change angular velocity
            actuator_space_info = { # actuators +/- (dx, dy,)
                        'dim': self.a_dim, 
                        'low': (-self.DELTA_SPEED, -self.DELTA_SPEED), 
                        'high':( self.DELTA_SPEED,  self.DELTA_SPEED), }
        else:
            #   ~  set velocity
            actuator_space_info = { # actuators +/- (dx, dy)
                        'dim': self.a_dim, 
                        'low': (-self.SPEED_LIMIT, -self.SPEED_LIMIT ), 
                        'high':( self.SPEED_LIMIT,  self.SPEED_LIMIT ), }

        self.actuator_space = get_nspace(n=self.N_BOTS, dtype=self.ACTION_DTYPE, 
                    shape = (actuator_space_info['dim'], ) ,
                    low=    actuator_space_info['low'],
                    high=   actuator_space_info['high'])
        self.base_actuator = np.zeros(self.actuator_space.shape, self.actuator_space.dtype)
        self.actuator = self.base_actuator.reshape((self.N_BOTS, self.a_dim ))
        #self.initial_actuator = np.zeros_like(self.actuator)

        # define action views
        self.δxy = self.actuator     [:, 0:2] 
        self.δx = self.actuator     [:, 0:1] 
        self.δy = self.actuator     [:, 1:2] 

        self.action_space = get_nspace(n=self.N_BOTS, shape=(self.a_dim,), dtype=self.ACTION_DTYPE, low=-1, high=1)
        self.action_mapper = REMAP(
                Input_Range=  ( self.action_space.low, self.action_space.high ), 
                Mapped_Range=( self.actuator_space.low, self.actuator_space.high ))
        return

    def build_vectors(self):
        # extra structures required
        self.dmat = np.zeros((self.N_BOTS, self.N_BOTS), dtype=self.STATE_DTYPE)
        self.alive = np.zeros((self.N_BOTS,), dtype=np.bool8 )
        # sensor images
        if self.enable_imaging:
            self.img_oray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)         
            self.img_xray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16) 
            self.img_dray =  np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)
        
        # sensor data
        sensor_data_keys = ( 'avail', 'xpos', 'ypos', 'xneg', 'yneg', 'x', 'y', 'difx', 'dify', 'dis', 'arch', 'arcl', 'arcm', 'occuded' )
        self.sensor_data = np.zeros((self.N_BOTS,  self.N_BOTS, len(sensor_data_keys) ), dtype=self.STATE_DTYPE)  



        return

    def build_render_modes(self):
        self.render_modes = {
            'all':      lambda : [self.render()],
            'env':      lambda : [self.render_env()],
            'rew':      lambda : [self.render_reward_signal()],
            'xray':     lambda : [self.render_sensor_image_global(True)],
            'dray':     lambda : [self.render_sensor_image_global(False)], 
            'xray_':    lambda : [self.render_sensor_image_local(n, True) for n in range(self.N_BOTS)]   ,
            'dray_':    lambda : [self.render_sensor_image_local(n, False) for n in range(self.N_BOTS)]   ,
            'sen_':     lambda : [self.render_sensor_data_local(n) for n in range(self.N_BOTS)]   ,
            }

    def build_meta(self,  xray_cmap='hot', dray_cmap='copper', record_reward_hist=True, show_plots=True,
                    render_figure_ratio=0.8, render_bounding_width=0.05, render_sensor_figure_size=10 ):
        self.sensor_meta_xray = {'cmap': xray_cmap, 'vmin': 0, 'vmax': self.N_BOTS}
        self.sensor_meta_dray = {'cmap': dray_cmap, 'vmin': 0, 'vmax': self.SCAN_RADIUS}
        self.render_figure_ratio = render_figure_ratio
        self.render_bounding_width = render_bounding_width
        self.render_sensor_figure_size = render_sensor_figure_size
        self.record_reward_hist = record_reward_hist
        self.show_plots = show_plots
        return



    """ Section: State Dynamics """

    def update(self):
        self.update_distances()
        self.update_alive()
        self.update_sensor_data()

    def update_distances(self):
        # first - update distances
        for f in range(self.N_BOTS):
            if self.alive[f]:
                self.dmat[f,f]=0.0
                for t in range(f+1, self.N_BOTS):
                    di = (np.linalg.norm(self.xy[f] - self.xy[t], 2) if self.alive[t] else 0.0)
                    self.dmat[f,t] = di
                    self.dmat[t,f] = self.dmat[f,t]
            else:
                self.dmat[f,:]=0.0
                self.dmat[:,f]=0.0
    def update_alive(self):
        # check bots <-- do not merge with above loop
        for f in range(self.N_BOTS):
            if not self.alive[f]:
                continue
            this_dis = self.dmat[f] # check where this bots' distance is less than diameter of other robots
            if len (np.where( (this_dis<=2*self.BOT_RADIUS) & (this_dis>0) )[0])>0 or  ( 
                (self.x[f,0]    <   -self.X_RANGE   ) or \
                (self.x[f,0]    >   self.X_RANGE    ) or \
                (self.y[f,0]    <   -self.Y_RANGE   ) or \
                (self.y[f,0]    >   self.Y_RANGE    ) \
                ):
                self.alive[f] = False
                self.n_alive-=1
                
                # clean up
                if self.enable_imaging:
                    self.img_xray[f,:]=0
                    self.img_oray[f,:]=0
                    self.img_dray[f,:]=self.SCAN_RADIUS
                
                self.sensor_data[f, :, :] = 0 
                #print(f'Robot Terminated: {self.names[f]} :: {self.n_alive} Robots left.')
        self.a[:, 0] = self.alive
        return
    def update_sensor_data(self):
        for b in range(self.N_BOTS):
            if not self.alive[b]:
                continue

            this_dis = self.dmat[b] # check where this bots' distance is less than diameter of other robots
            # we want neighbours in sorted order according to their distance
            this_argsort = this_dis.argsort()
            this_dis_view = this_dis[this_argsort]
            this_indices = np.where( 
                (this_dis_view<=self.SCAN_RADIUS) & \
                ((this_dis_view>0)) & \
                (self.alive[this_argsort]==True) )[0] # neighbours
            N = this_argsort[this_indices]
            nos_N = len(N)
            # clean up ray data before scanning afresh
            if self.enable_imaging:
                self.img_xray[b,:]=0
                self.img_oray[b,:]=0
                self.img_dray[b,:]=self.SCAN_RADIUS
            self.sensor_data[b, :, : ]=0

            for i,n in enumerate(N):
                assert(self.alive[n])
                difference = self.xy[n, :] - self.xy[b, :]
                distance = self.dmat[b, n]
                boundary = (self.SCAN_RADIUS/distance)*difference
                theta = np.arcsin( self.BOT_RADIUS/distance )

                difx = difference[0]
                dify = difference[1]

                x = boundary[0] 
                y = boundary[1] 

                xpos = x*np.cos(theta) - y*np.sin(theta)   
                ypos = x*np.sin(theta) + y*np.cos(theta)    

                xneg = x*np.cos(-theta) - y*np.sin(-theta)
                yneg = x*np.sin(-theta) + y*np.cos(-theta) 

                arch = int(get_angle( (xpos, ypos) ) * self.SENSOR_UNIT_ARC)
                arcm = int(get_angle( (x, y) ) * self.SENSOR_UNIT_ARC)
                arcl = int(get_angle( (xneg, yneg) ) * self.SENSOR_UNIT_ARC)

                if self.enable_imaging:
                    img = self.img_dray[b,:]
                    self.img_oray[b, arcm%self.SENSOR_IMAGE_SIZE]+=1
                    if arcl>arch:
                        #arcl, arch = arch, arcl
                        self.img_xray[b, 0:arch]+=1
                        self.img_xray[b, arcl: ]+=1
                        self.img_dray[b, np.where(img[0:arch]>distance)[0]] = distance
                        self.img_dray[b, arcl+np.where(img[arcl:]>distance)[0]] = distance
                    else:
                        self.img_xray[b, arcl:arch]+=1
                        self.img_dray[b, arcl+np.where(img[arcl:arch]>distance)[0]] = distance
                                                # 0  1     2     3     4      5  6  7     8     9         10     11   12    13
                self.sensor_data[b, i, : ] = (1, xpos, ypos, xneg, yneg,  x, y, difx, dify, distance, arch, arcl, arcm, 0)
            # ---> all neighbours end
            #distance_key = 9
            arch_key, arcl_key = 10, 11
            self.occn[b,:] = 0
            self.alln[b,:] = nos_N

            for ni in range(1, nos_N):
            # the nearest bot is already sorted
                # if lower bots are occulded or not
                zi = ni-1
                while zi>=0:
                    ah, al = self.sensor_data[b, ni, arch_key], self.sensor_data[b, ni, arcl_key]
                    if ah<al:
                        ah += self.SENSOR_IMAGE_SIZE

                    zh, zl = self.sensor_data[b, zi, arch_key], self.sensor_data[b, zi, arcl_key]
                    if zh<zl:
                        zh += self.SENSOR_IMAGE_SIZE

                    #first determine lower
                    if al>zl:
                        zl, zh, al, ah = al, ah, zl, zh
                    if not (al<zl and ah<zh and ah<zl):
                        self.sensor_data[b, ni, -1] = 1
                        self.occn[b,:]+=1
                        zi=-1
                    else:
                        zi-=1

        
        # ------------------> end for all bots
        
        return

    def is_done(self):
        return bool( (self.ts>=self._max_episode_steps) or ( self.n_alive > 0)  )

    def reset(self):
        # reset - choose randomly from known initial state distribution - state numbers start at 1
        self.choose_i_state = self.rng.integers(0, self.initial_states_count)
        for i,p in  enumerate(self.initial_states[self.choose_i_state]):
            self.initial_observation[i, 0:2] = p # p = (x,y) 2-tuple


        self.actuator*=0 # reset actuator
        self.observation[:] = self.initial_observation # copy state vectors
        self.alive[:]=True # all robots alive flag
        self.n_alive = np.sum(self.alive)
        

        # update state variables
        self.update()
        self.reward_signal[:] = self.get_reward_signal()
        self.reward = 0.0
        self.ts=0
        self.done=self.is_done()
        
        self.reward_hist = [[np.sum(self.reward_signal)], [self.reward], [0.0]] # signal sum, current reward, cumm reward
        self.cummulative_reward = 0.0
        return self.base_observation
    
    def step(self, action):
        self.base_actuator[:] = self.action_mapper.in2map(np.clip( action, self.action_space.low, self.action_space.high )) # copy actions
        self.actuator[np.where(self.alive==False)[0],:]=0 # not act on crahsed robots

        # set actuator values
        if self.delta_action_mode:
            self.dxy[:] = np.clip(self.dxy+self.δxy, -self.SPEED_LIMIT, self.SPEED_LIMIT)
        else:
            self.dxy[:] = self.δxy
        # assume that veclocities are updates - x,y,, dx,dy
        self.xy+=self.dxy # move forward - unrestricted
        self.update() # distance matrix updated
        self.ts+=1
        self.done=self.is_done()
        if not self.done:
            # reward is based on wighted signal
            current_reward_signal = self.get_reward_signal()
            # if current reward_signal is higher, then give +ve reward
            self.reward = float(np.sum(current_reward_signal - self.reward_signal)) #np.sum((self.rsA - rsA) * self.rw8)
            self.reward_signal[:] = current_reward_signal
        else:
            self.reward=-1.0
        
        self.cummulative_reward += self.reward
        if self.record_reward_hist:
            self.reward_hist[0].append(np.sum(self.reward_signal))
            self.reward_hist[1].append(self.reward)
            self.reward_hist[2].append(self.cummulative_reward)
        return self.base_observation, self.reward, self.done, {}
    

    """ Section: Reward Signal : implement in inherited classes """

    def set_reward_signal(self):        
        # by default creates a -ve and a +ve reward signal and generates randomly
        self.reward_labels =          ['+ve reward', '-ve reward']
        self.reward_sign =  np.array( [      1,           -1     ], dtype=self.REWARD_DTYPE) # <--- +/- reward
        self.reward_norms = np.array( [     10,           10     ], dtype=self.REWARD_DTYPE) # <--- maximum value of Reward

    def build_reward_signal(self):
        self.reward_w8 = self.reward_sign/self.reward_norms #<---- this gets multiplied by reward from environment
        self.reward_signal_n = len(self.reward_labels)
        self.reward_plot_x = np.arange(self.reward_signal_n)
        self.reward_posw8 = np.where(self.reward_sign>0)[0]
        self.reward_negw8 = np.where(self.reward_sign<0)[0]
        self.reward_signal = np.zeros_like(self.reward_w8)
        self.reward_limits = self.reward_sign * self.reward_norms
        # for recording reward hist
        self.reward_rng = np.random.default_rng(None)
        
    def get_reward_signal(self):  # higher is better
        # return self.reward_w8 * np.array(  ('n-tuple, no of reward signals, max at self.reward_norms') ,  dtype=self.REWARD_DTYPE)
        return self.reward_w8 *  (self.reward_rng.random(size=(self.reward_signal_n,))*self.reward_norms)
        
        
          

    """ Section: Sensor Rendering """

    def render_sensor_image_local(self, n, use_xray=False, show_ticks=True):
        if not self.enable_imaging:
            return None, ""
        fig,_ = plt.figure()
        if show_ticks:
            plt.xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
            
            plt.grid(axis='both')

        if use_xray:
            plt.imshow(np.reshape( (self.img_xray[n, :]), (1, self.SENSOR_IMAGE_SIZE) ), aspect=self.img_aspect, **self.sensor_meta_xray)
            arcm = np.where(self.img_oray[n]>0)[0]
            for arcpt in arcm:
                plt.scatter( [arcpt], [0], color='white', marker='d') # self.img_oray[n,arcpt:arcpt+1]
                # f'${self.img_oray[n,arcpt]}$'
                plt.annotate(f'{self.img_oray[n,arcpt]}', xy=(arcpt,0.4))
            plt.title("X-Ray: "+self.names[n])
        else:
            plt.imshow(np.reshape( (self.img_dray[n, :]), (1, self.SENSOR_IMAGE_SIZE) ), aspect=self.img_aspect, **self.sensor_meta_dray)
            plt.title("D-Ray: "+self.names[n])
        
        
        (plt.show() if self.show_plots else plt.close())
        return fig, "render_sensor_image_local"

    def render_sensor_image_global(self, use_xray=False, show_ticks=True):
        if not self.enable_imaging:
            return None, ""
        # renders the image seen by all sensors - oray, xday, dray
        fig,_ = plt.figure(figsize=( (self.render_sensor_figure_size, self.N_BOTS) ))
        if show_ticks:
            plt.yticks(range(self.N_BOTS), self.names)
            plt.xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
            
            plt.grid(axis='both')
            
        #plt.title()
        if use_xray:
            plt.imshow(self.img_xray, aspect=self.img_aspect, **self.sensor_meta_xray)
            plt.title("X-Ray: All Sensors")
            #for n in range(self.N_BOTS):
            #    plt.scatter( np.arange(self.SENSOR_IMAGE_SIZE), self.img_oray[n], color=self.colors[n])
        else:
            plt.imshow(self.img_dray, aspect=self.img_aspect, **self.sensor_meta_dray)
            plt.title("D-Ray: All Sensors")
        
        (plt.show() if self.show_plots else plt.close())
        return fig, "render_sensor_image_global"

    def render_sensor_data_local(self, n):
        if not self.alive[n]:
            return None, ""
        fig,ax = plt.subplots(1, 1, figsize=(self.render_sensor_figure_size, self.render_sensor_figure_size))
        ax.set_title("SENSOR MAP: "+self.names[n])
        limL, limH = -self.SCAN_RADIUS*1.25, self.SCAN_RADIUS*1.25
        ax.set_xlim(limL, limH)
        ax.set_ylim(limL, limH)
        plt.scatter([0], [0], color=self.colors[n], marker = self.markers[n])
        ax.add_patch( # speedometer
            Circle(   ( 0, 0 ), ( self.BOT_RADIUS),    # 2-tuple center, # float radius
                color=self.colors[n], linewidth=2.0, fill=False))
        ax.add_patch( # speedometer
            Circle(   ( 0, 0 ), ( self.SCAN_RADIUS),    # 2-tuple center, # float radius
                color=self.colors[n] if self.alive[n] else 'black', linewidth=0.5, fill=False))
        ax.add_patch( # safe distance
            Circle(   ( 0, 0 ), ( self.SAFE_CENTER_DISTANCE -self.BOT_RADIUS),    # 2-tuple center, # float radius
                color='black', linewidth=2.0, fill=False, linestyle='dotted'))
    
    

        neighbours = self.sensor_data [ n, np.where(self.sensor_data[n, :, 0]>0)[0], :]
        ax.set_xlabel(f'Total: {self.alln[n, 0]:.0f}, Occluded: {self.occn[n, 0]:.0f}')
        for g in neighbours:
            xp,     yp,     xn,     yn,  x,     y,      dx,     dy,   occ = \
            g[1],   g[2],   g[3],  g[4], g[5],  g[6],   g[7],   g[8],  g[-1]
            #(1, xpos, ypos, xneg, yneg,  x, y, difx, dify, distance, arch, arcl, arcm, occluded)
            

            # the shadow arcs vectors and tangesnts
            #ax.annotate(str((180/np.pi)*get_angle((x,y))) , xy=(x,y))
            line_style_arc = ('dashed' if occ else 'solid')
            ax.annotate("",  xytext=(0, 0), xy=( 0 + x, 0 + y ),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc , color='black')) # distance vector
            ax.annotate("", xytext=(0, 0), xy=(0 + xn, 0 +yn ),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc, color="tab:red")) # neg arc
            ax.annotate("",  xytext=(0, 0), xy=( 0 + xp, 0 + yp),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc, color="tab:green")) # pos arc


            # vectors
            line_style_vec = ('dotted' if occ else 'solid')
            ax.annotate("", xytext=(0, 0), xy=( 0 + dx, 0 + dy ),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color='black'))
            ax.annotate("", xytext=(x, y), xy=(0 + xn, 0 +yn ),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color="tab:red"))
            ax.annotate("", xytext=(x, y), xy=( 0 + xp, 0 + yp),
                arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color="tab:green"))

            ax.add_patch( 
                Circle(   ( dx, dy ), ( self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color=('tab:grey' if occ else 'black') , linewidth=2.0, fill=False, linestyle=line_style_vec))
            

        (plt.show() if self.show_plots else plt.close())
        return fig, "render_sensor_data_local"

    """ Section: Global Rendering """

    def render_reward_signal(self):
        fig,_ = plt.figure(figsize=( self.reward_signal_n , 5 ))
        plt.ylim(0.0,1.1) 
        plt.bar(self.reward_posw8, self.reward_signal[self.reward_posw8], color='tab:green' )
        plt.bar(self.reward_negw8, -self.reward_signal[self.reward_negw8], color='tab:red' )
        plt.xticks(self.reward_plot_x, self.reward_labels)
        for i in range(self.reward_signal_n):
            plt.annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.25, 1.1)  )
        for p in self.reward_posw8:
            plt.vlines(p, 0, 1, color='green')
        for p in self.reward_negw8:
            plt.vlines(p, 0, 1, color='red')
        plt.title('Reward Signal')
        (plt.show() if self.show_plots else plt.close())
        return fig, "render_reward_signal"

    def render_env(self):

        fig = plt.figure(figsize=((self.X_RANGE)*2*self.render_figure_ratio,(self.Y_RANGE)*2*self.render_figure_ratio))
        plt.title(f'{self.name}/{self.choose_i_state} :: Step-{self.ts}')
        ax = fig.axes[0]

        # bounding box width 
        ax.set_xlim((-self.X_RANGE-int(self.X_RANGE*self.render_bounding_width), self.X_RANGE+int(self.X_RANGE*self.render_bounding_width)))
        ax.set_ylim((-self.Y_RANGE-int(self.Y_RANGE*self.render_bounding_width), self.Y_RANGE+int(self.Y_RANGE*self.render_bounding_width)))
        ax.vlines(-self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed')
        ax.vlines(self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(-self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        if self.TARGET_RADIUS>0:
            ax.add_patch( # body (bot radius)
                Circle(   ( 0, 0 ), ( self.TARGET_RADIUS ),    # 2-tuple center, # float radius
                    color='black', linewidth=0.5, fill=False, linestyle='dashed'))
        for n in range(self.N_BOTS):

        #--------------------------------------------------------------------------------------------------------------
            botx, boty = self.x[n, 0], self.y[n, 0]
            botcolor = self.colors[n]
            botmarker = self.markers[n]
            bot_speed = np.linalg.norm(self.dxy[n], 2)

            #--------------------------------------------------------------------------------------------------------------
            ax.vlines(0, -self.Y_RANGE, self.Y_RANGE, color='black', linewidth=0.8, linestyle='dashed' )
            ax.hlines(0, -self.X_RANGE, self.X_RANGE, color='black', linewidth=0.8, linestyle='dashed')
            #--------------------------------------------------------------------------------------------------------------
            ax.scatter( [botx], [boty], color=botcolor, marker=botmarker )
            if  not self.alive[n]:
                botcolor='tab:gray'
            ax.add_patch( # body (bot radius)
                Circle(   ( botx, boty ), ( self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=1.0, fill=False, linestyle='solid'))
            ax.add_patch( # safe distance
                Circle(   ( botx, boty ), ( self.SAFE_CENTER_DISTANCE-self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color='black', linewidth=1.0, fill=False, linestyle='dotted'))

            ax.add_patch( # radar (scan radius)
                Circle(   ( botx, boty ), ( self.SCAN_RADIUS ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=0.5, fill=False, linestyle='dashed'))
            ax.add_patch( # speedometer
                Circle(   ( botx, boty ), ( bot_speed*self.BOT_RADIUS/self.MAX_SPEED ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=1.0, fill=True))

            velocity = self.xy[n] + ((self.dxy[n]/bot_speed) if bot_speed!=0 else 0)
            ax.annotate("", 
                #xytext="Face Vectors = sum of dx and dy",
                xytext=(botx, boty), xy=( velocity[0], velocity[1] ),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, linestyle='solid', color='black'))
            
            #--------------------------------------------------------------------------------------------------------------
            if  self.alive[n]:
        
                neighbours = self.sensor_data [ n, np.where(self.sensor_data[n, :, 0]>0)[0], :]
                #ax.set_title(str(len(neighbours)))
                for g in neighbours:
                    xp,     yp,     xn,     yn,  x,     y     = \
                    g[1],   g[2],   g[3],  g[4], g[5],  g[6]
                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=( botx + x, boty + y ),
                        arrowprops=dict(arrowstyle="->", linewidth=0.8, linestyle='dashed', color='black'))

                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=(botx + xn, boty +yn ),
                        arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=( botx + xp, boty + yp),
                        arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
        #--------------------------------------------------------------------------------------------------------------

        (plt.show() if self.show_plots else plt.close())
        return fig, "render_env"

    def render(self):

        fig = plt.figure(
            constrained_layout=True,
            figsize=(   (self.X_RANGE)*2*self.render_figure_ratio *4/3 ,
                        (self.Y_RANGE)*2*self.render_figure_ratio   )
            )
        subfigs = fig.subfigures(1, 2, wspace=0.02, width_ratios=[3, 1]) # row, col

        subfigs[0].suptitle(f'{self.name}/{self.choose_i_state} :: Step-{self.ts}')
        ax = subfigs[0].subplots(1, 1)
        #subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

        # bounding box width 
        ax.set_xlim((-self.X_RANGE-int(self.X_RANGE*self.render_bounding_width), self.X_RANGE+int(self.X_RANGE*self.render_bounding_width)))
        ax.set_ylim((-self.Y_RANGE-int(self.Y_RANGE*self.render_bounding_width), self.Y_RANGE+int(self.Y_RANGE*self.render_bounding_width)))
        ax.vlines(-self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed')
        ax.vlines(self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(-self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        if self.TARGET_RADIUS>0:
            ax.add_patch( # body (bot radius)
                Circle(   ( 0, 0 ), ( self.TARGET_RADIUS ),    # 2-tuple center, # float radius
                    color='black', linewidth=0.5, fill=False, linestyle='dashed'))
        for n in range(self.N_BOTS):

        #--------------------------------------------------------------------------------------------------------------
            botx, boty = self.x[n, 0], self.y[n, 0]
            botcolor = self.colors[n]
            botmarker = self.markers[n]
            bot_speed = np.linalg.norm(self.dxy[n], 2)

            #--------------------------------------------------------------------------------------------------------------
            ax.vlines(0, -self.Y_RANGE, self.Y_RANGE, color='black', linewidth=0.8, linestyle='dashed' )
            ax.hlines(0, -self.X_RANGE, self.X_RANGE, color='black', linewidth=0.8, linestyle='dashed')
            #--------------------------------------------------------------------------------------------------------------
            ax.scatter( [botx], [boty], color=botcolor, marker=botmarker )
            if  not self.alive[n]:
                botcolor='tab:gray'
            ax.add_patch( # body (bot radius)
                Circle(   ( botx, boty ), ( self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=1.0, fill=False, linestyle='solid'))
            ax.add_patch( # safe distance
                Circle(   ( botx, boty ), ( self.SAFE_CENTER_DISTANCE-self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color='black', linewidth=1.0, fill=False, linestyle='dotted'))

            ax.add_patch( # radar (scan radius)
                Circle(   ( botx, boty ), ( self.SCAN_RADIUS ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=0.5, fill=False, linestyle='dashed'))
            ax.add_patch( # speedometer
                Circle(   ( botx, boty ), ( bot_speed*self.BOT_RADIUS/self.MAX_SPEED ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=1.0, fill=True))

            velocity = self.xy[n] + ((self.dxy[n]/bot_speed) if bot_speed!=0 else 0)
            ax.annotate("", 
                #xytext="Face Vectors = sum of dx and dy",
                xytext=(botx, boty), xy=( velocity[0], velocity[1] ),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, linestyle='solid', color='black'))
            
            #--------------------------------------------------------------------------------------------------------------
            if  self.alive[n]:
        
                neighbours = self.sensor_data [ n, np.where(self.sensor_data[n, :, 0]>0)[0], :]
                #ax.set_title(str(len(neighbours)))
                for g in neighbours:
                    xp,     yp,     xn,     yn,  x,     y     = \
                    g[1],   g[2],   g[3],  g[4], g[5],  g[6]
                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=( botx + x, boty + y ),
                        arrowprops=dict(arrowstyle="->", linewidth=0.8, linestyle='dashed', color='black'))

                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=(botx + xn, boty +yn ),
                        arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=( botx + xp, boty + yp),
                        arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
        #--------------------------------------------------------------------------------------------------------------

        #bx = subfigs[1].subplots(3, 1, width_ratios=[4, 1, 1])
        bx = subfigs[1].subplots(6, 1, gridspec_kw={'height_ratios': [1, 1, 4, 1, 1,1]})
       # plt.figure(figsize=( self.reward_signal_n , 5 ))
        bx[2].set_ylim(0.0,1.1)
        bx[2].bar(self.reward_posw8, self.reward_signal[self.reward_posw8], color='tab:green' )
        bx[2].bar(self.reward_negw8, -self.reward_signal[self.reward_negw8], color='tab:red' )
        bx[2].set_xticks(self.reward_plot_x, self.reward_labels)
        for i in range(self.reward_signal_n):
            bx[2].annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.15, 1.05)  )
        for p in self.reward_posw8:
            bx[2].vlines(p, 0, 1, color='green')
        for p in self.reward_negw8:
            bx[2].vlines(p, 0, 1, color='red')
        bx[2].set_title('Reward Signal')

        bx[3].plot(self.reward_hist[1], color='tab:green')
        bx[3].set_title(f'Step-Reward : {self.reward_hist[1][-1]:.3f}')

        bx[4].plot(self.reward_hist[2], color='tab:brown')
        bx[4].set_title(f'Cummulative-Reward : {self.reward_hist[2][-1]:.3f}')
        
        bx[5].plot(self.reward_hist[0], color='tab:blue')
        bx[5].set_title(f'Signal Sum : {self.reward_hist[0][-1]:.3f}')

        if self.enable_imaging:
        
            #plt.figure(figsize=( (self.render_sensor_figure_size, self.N_BOTS) ))
            bx[1].set_yticks(range(self.N_BOTS), self.names)
            bx[1].set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
            bx[1].grid(axis='both')
            bx[1].imshow(self.img_xray, aspect=self.img_aspect, **self.sensor_meta_xray)
            bx[1].set_title("X-Ray: All Sensors")
            
            bx[0].set_yticks(range(self.N_BOTS), self.names)
            bx[0].set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
            bx[0].grid(axis='both')
            bx[0].imshow(self.img_dray, aspect=self.img_aspect, **self.sensor_meta_dray)
            bx[0].set_title("D-Ray: All Sensors")
        

        
        (plt.show() if self.show_plots else plt.close())
        return fig, "render"

#==============================================================

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
""" NOTE: This file should contain the core World class only, all inherited classes are in models.py
"""
#--------------------------------------------------------------------------------------------------------------