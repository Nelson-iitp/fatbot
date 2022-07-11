#==============================================================
import numpy as np
from math import ceil, inf, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import gym, gym.spaces
from io import BytesIO
import os, cv2
from enum import IntEnum
from .common import get_nspace, get_angle, REMAP
#==============================================================

class Swarm:
    default_bot_colors = [ 'red', 'blue', 'green', 'gold',   'cyan', 'magenta', 'purple', 'brown' ]
    default_n_bots = len(default_bot_colors)
    default_bot_markers = ['o' for _ in range(default_n_bots)]
    def __init__(self, 
                    name,               # identifier
                    x_range,            # ranges from +x to -x
                    y_range,            # ranges from +y to -y
                    n_bots,             # number of bots in swarm
                    bot_radius,         # meters - fat-bot body radius
                    scan_radius,        # scannig radius of onboard sensor (neighbourhood)
                    safe_distance,      # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
                    speed_limit,        # [+/-] upper speed (roll, pitch) limit of all robots
                    delta_speed,        # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
                    sensor_resolution,  # choose based on scan distance, use form: n/pi , pixel per m
                    min_bots_alive,     # min no. bots alive, below which world will terminate # keep zero for all alive condition
                    horizon,            # max timesteps
                    target_radius=0.0,  # optinal
                ) -> None:
        assert ((n_bots<=self.default_n_bots) and (n_bots>0))
        self.name, self.x_range, self.y_range, self.horizon =  \
             name,      x_range,      y_range,      horizon 
        self.n_bots, self.bot_radius, self.scan_radius, self.safe_distance, self.speed_limit, self.delta_speed, self.sensor_resolution = \
             n_bots,      bot_radius,      scan_radius,      safe_distance,      speed_limit,      delta_speed,      sensor_resolution    
        self.bot_colors = self.default_bot_colors[0:self.n_bots]
        self.bot_names = self.default_bot_colors[0:self.n_bots]
        self.bot_markers = self.default_bot_markers[0:self.n_bots]
        self.min_bots_alive = min_bots_alive
        self.target_radius = target_radius
        self.initial_states= [] #<--- after init, append to this list

class World(gym.Env):
    
    # Data-types
    STATE_DTYPE =    np.float32
    ACTION_DTYPE =   np.float32
    REWARD_DTYPE =   np.float32

    def info(self):
        return f'{self.name} \n Dim: ( X={self.X_RANGE*2}, Y={self.Y_RANGE*2}, H={self._max_episode_steps} ),  Imaging: [{self.enable_imaging}],  History: [{self.record_reward_hist}]'
    
    def __init__(self, swarm, enable_imaging=True, horizon=0, seed=None, custom_XY=None, 
                    record_reward_hist=True, render_normalized_reward=True,
                    render_xray_cmap='hot', render_dray_cmap='copper',  render_dpi=None,
                    render_figure_ratio=1.0, render_bounding_width=0.05) -> None:
        super().__init__()
        
        self.swarm = swarm
        self.enable_imaging = enable_imaging
        self._max_episode_steps = (horizon if horizon>0 else inf)
        self.record_reward_hist = record_reward_hist
        self.render_normalized_reward = render_normalized_reward
        self.rng = np.random.default_rng(seed)

        if custom_XY is None:
            self.X_RANGE, self.Y_RANGE = float(self.swarm.x_range), float(self.swarm.y_range)
        else:
            self.X_RANGE, self.Y_RANGE = float(custom_XY[0]), float(custom_XY[1])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.build() # call default build
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # counter
        self.episode = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # render related
        self.name = f'-({self.swarm.name})-'
        self.render_xray_cmap = render_xray_cmap
        self.render_dray_cmap = render_dray_cmap
        self.render_figure_ratio = render_figure_ratio
        self.render_bounding_width = render_bounding_width
        self.render_dpi = render_dpi

        # for rendering
        self.MAX_SPEED = sqrt(2) * self.SPEED_LIMIT # note - this is magnitude of MAX velocity vector
        self.colors = self.swarm.bot_colors
        self.markers = self.swarm.bot_markers
        self.names = self.swarm.bot_names
        self.TARGET_RADIUS = swarm.target_radius
        self.img_aspect = self.SENSOR_IMAGE_SIZE/self.SENSOR_RESOULTION # for rendering
        self.arcDivs = 4 + 1  # no of divisions on sensor image
        self.arcTicks = np.array([ (int( i * self.SENSOR_UNIT_ARC), round(i*180/pi, 2)) \
                                    for i in np.linspace(0, 2*np.pi, self.arcDivs) ])
        print(f'[*] World Created :: {self.info()}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    """ Section: Build """

    def build(self):
        # --> call in order
        self.build_params()
        self.build_observation_space()
        self.build_action_space()
        self.build_vectors()
        self.build_reward_signal()
        

    def build_params(self):
        # swarm
        self.N_BOTS = int(self.swarm.n_bots)
        self.BOT_RADIUS = float(self.swarm.bot_radius)
        self.SCAN_RADIUS = float(self.swarm.scan_radius)
        self.SAFE_CENTER_DISTANCE = float(self.swarm.safe_distance) + self.BOT_RADIUS * 2
        self.SPEED_LIMIT = float(self.swarm.speed_limit)
        self.DELTA_SPEED = float(self.swarm.delta_speed)
        self.delta_action_mode = (self.DELTA_SPEED!=0.0)
        self.SENSOR_RESOULTION = float(self.swarm.sensor_resolution)
        self.SENSOR_UNIT_ARC = self.SENSOR_RESOULTION * self.SCAN_RADIUS
        self.SENSOR_IMAGE_SIZE = int(ceil(2 * np.pi * self.SENSOR_UNIT_ARC)) # total pixels in sensor data input
        self.MIN_BOTS_ALIVE = (self.swarm.min_bots_alive if self.swarm.min_bots_alive>0 else self.N_BOTS)
        self.initial_states = self.swarm.initial_states
        self.initial_states_count = len(self.initial_states)
        
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
        neighbour_space_info = { # total_neighbours, # occluded neighbours
                        'dim': 2,  
                        'low': (0,           0,         ), 
                        'high':(self.N_BOTS, self.N_BOTS), 
                    }
        self.observation_space = get_nspace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
                    shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'], ),
                    low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low']  ),
                    high=   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] ))
        
        self.o_dim =6
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
        # sensor data
        sensor_data_keys = ( 'avail', 'xpos', 'ypos', 'xneg', 'yneg', 'x', 'y', 'difx', 
                                'dify', 'dis', 'arch', 'arcl', 'arcm', 'occuded' )
        self.sensor_data = np.zeros((self.N_BOTS,  self.N_BOTS, len(sensor_data_keys) ), dtype=self.STATE_DTYPE)  
        self.sensor_avail = self.sensor_data[:, :, 0]
        
        # sensor images
        if self.enable_imaging:
            self.img_oray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)         
            self.img_xray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16) 
            self.img_dray =  np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)
        return



    """ Section: State Dynamics """

    def kill_bots(self, *bots):
        for b in bots:
            self.alive[b] = False
            self.dmat[b,:]=0
            self.dmat[:,b]=0
            self.sensor_data[b, :, :] = 0 
            if self.enable_imaging:
                self.img_xray[b,:]=0
                self.img_oray[b,:]=0
                self.img_dray[b,:]=self.SCAN_RADIUS
        self.n_alive = np.sum(self.alive)

    def update_distances(self):
        for f in range(self.N_BOTS):
            if self.alive[f]:
                #self.dmat[f,f]=0.0
                if ( 
                    (self.x[f,0]    <   -self.X_RANGE   ) or \
                    (self.x[f,0]    >   self.X_RANGE    ) or \
                    (self.y[f,0]    <   -self.Y_RANGE   ) or \
                    (self.y[f,0]    >   self.Y_RANGE    ) \
                ):
                    self.kill_bots(f)
                else:
                    for t in range(f+1, self.N_BOTS):
                        #di = (np.linalg.norm(self.xy[f] - self.xy[t], 2) if self.alive[t] else 0.0)
                        di = np.linalg.norm(self.xy[f] - self.xy[t], 2)
                        self.dmat[f,t] = di
                        self.dmat[t,f] = self.dmat[f,t]
                        if di<=2*self.BOT_RADIUS:
                            self.kill_bots(f,t)

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
                #assert(self.alive[n])
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
        return bool( (self.ts>=self._max_episode_steps) or ( self.n_alive < self.MIN_BOTS_ALIVE)  )

    def reset(self, starting_state=None):
        self.episode += 1
        # reset - choose randomly from known initial state distribution - state numbers start at 1
        if starting_state is None:
            self.choose_i_state = self.rng.integers(0, self.initial_states_count)
        else:
            self.choose_i_state = int(starting_state(self.episode))

        for i,p in  enumerate(self.initial_states[self.choose_i_state]):
            self.initial_observation[i, 0:2] = p # p = (x,y) 2-tuple


        self.actuator*=0 # reset actuator
        self.observation[:] = self.initial_observation # copy state vectors
        self.alive[:]=True # all robots alive flag
        self.n_alive = np.sum(self.alive)
        

        # update state variables----------------------
        self.update_distances()
        self.update_sensor_data()
        #---------------------------------------------

        self.reward_signal[:] = self.get_reward_signal()
        self.reward_signal_sum = np.sum(self.reward_signal)
        self.step_reward = 0.0
        self.cummulative_reward = 0.0
        self.ts=0
        self.done=self.is_done()
        if self.record_reward_hist:
            self.reward_hist = [
                [self.reward_signal_sum ], 
                [self.step_reward], 
                [self.cummulative_reward]
                ] # signal sum, current reward, cumm reward
        
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

        # update state variables----------------------
        self.update_distances()
        self.update_sensor_data()
        #---------------------------------------------

        self.ts+=1
        self.done=self.is_done()
        if not self.done:
            # reward is based on wighted signal
            current_reward_signal = self.get_reward_signal()
            current_reward_signal_sum = np.sum(current_reward_signal)
            # if current reward_signal is higher, then give +ve reward
            self.step_reward = float(current_reward_signal_sum - self.reward_signal_sum) #np.sum((self.rsA - rsA) * self.rw8)
            self.reward_signal[:] = current_reward_signal
            self.reward_signal_sum = current_reward_signal_sum
        else:
            self.step_reward=-1.0
        
        self.cummulative_reward += self.step_reward
        if self.record_reward_hist:
            self.reward_hist[0].append(self.reward_signal_sum)
            self.reward_hist[1].append(self.step_reward)
            self.reward_hist[2].append(self.cummulative_reward)
        return self.base_observation, self.step_reward, self.done, {}
    

    """ Section: Reward Signal : implement in inherited classes """

    def build_reward_scheme(self): 
        # by default creates a -ve and a +ve reward signal and generates randomly
        print('[!] Calling default build_reward_scheme function: Using Random Reward Signal')
        return dict(
            random_reward_pos = 1.0,
            random_reward_neg = 1.0
        )

    def build_reward_signal(self):

        reward_scheme = self.build_reward_scheme() 

        max_dis_bots = ((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS
        max_n_bots = (self.N_BOTS-1)*self.N_BOTS

        self.reward_data = dict(
                              #  sign,      low,      high              label
            random_reward_pos =    ( 1       -1,        1,                'R+'),
            random_reward_neg =    ( -1      -1,        1,                'R-'),

            # distance to target point <--- lower is better
            dis_target_point =    (    -1,         0,       max_dis_bots,     'C2P-Target',     ),

            # distance to target radius <--- lower is better
            dis_target_radius =    (    -1,         0,       max_dis_bots,     'C2R-Target',     ),

            # no of unsafe bots  <--- lower is better
            all_unsafe =    (    -1,         0,       max_n_bots,       'Safe-Bots',    ),

            # no of total neighbours  <--- higher is better
            all_neighbour = (     1,         0,       max_n_bots,       'Neighbours'    ),

            # no of occluded neighbours  <--- lower is better
            occluded_neighbour = (    -1,         0,       max_n_bots,       'V-Neighbours'  ),

            # occlusion ratio = occluded pixels / total pixels  <--- lower is better
            occlusion_ratio =     (    -1,         0,       1,                'V-Ratio'       ),

        )


        reward_labels, rsign, rlow, rhigh, rw8, reward_caller   = [], [], [], [], [], []
        for k,w in reward_scheme.items():
            v = self.reward_data[k]
            rsign.append( v[0] )
            rlow.append( v[1] )
            rhigh.append( v[2] )
            reward_labels.append( v[3] )
            rw8.append(w)
            reward_caller.append( getattr(self, f'RS_{k}') )

        self.reward_labels =   reward_labels
        self.rsign = np.array( rsign, dtype=self.REWARD_DTYPE) # <--- +(higher is better) /-(loweris better) 
        self.rlow = np.array(  rlow, dtype=self.REWARD_DTYPE) # <--- minimum value of Reward
        self.rhigh = np.array( rhigh, dtype=self.REWARD_DTYPE) # <--- maximum value of Reward
        self.rw8 = np.array(  rw8, dtype=self.REWARD_DTYPE)  #<---- this gets multiplied by reward from environment

        self.reward_caller=reward_caller
        if len (reward_caller)==0:
            print(f'[!] Reward Signal is empty!')

        #self.reward_posw8 = np.where(self.rsign>0)[0]
        reward_negw8 = np.where(self.rsign<0)[0]
        
        mapper_low = np.copy(self.rlow)
        mapper_low[reward_negw8] = self.rhigh[reward_negw8]

        mapper_high = np.copy(self.rhigh)
        mapper_high[reward_negw8] = self.rlow[reward_negw8]

        self.r_dim = len(self.reward_labels)
        
        self.reward_mapper = REMAP(
            Input_Range= (mapper_low,                mapper_high),
            Mapped_Range=(np.zeros_like(mapper_low), np.ones_like(mapper_high)))
        
        self.reward_signal = np.zeros_like(self.rw8)
        
        self.reward_plot_x = np.arange(self.r_dim) # for render
        self.reward_plot_y = np.max(self.rw8)
        self.reward_rng = np.random.default_rng(None) # for recording reward hist

    def get_reward_signal(self):  # higher is better
        return self.rw8 * self.reward_mapper.in2map(np.array( [ RF() for RF in self.reward_caller ], dtype=self.REWARD_DTYPE ))


    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Pre-defined reward signals """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    
    def RS_random_reward_pos(self):
        return self.reward_rng.uniform(-1, 1)

    def RS_random_reward_neg(self):
        return self.reward_rng.uniform(-1, 1)

    def RS_dis_target_point(self): 
        return np.sum ([ np.linalg.norm( self.xy[n, :], 2 )  for n in range(self.N_BOTS) ])

    def RS_dis_target_radius(self): 
        return np.sum ([ np.abs(self.TARGET_RADIUS-np.linalg.norm( self.xy[n, :], 2 ))  for n in range(self.N_BOTS) ])

    def RS_all_unsafe(self):
        return len(np.where(  (self.dmat<self.SAFE_CENTER_DISTANCE) & (self.dmat>0)  ) [0])

    def RS_all_neighbour(self):
        return np.sum(self.alln)

    def RS_occluded_neighbour(self):
        return np.sum(self.occn)

    def RS_occlusion_ratio(self):
        return np.sum ([  (len(np.where(self.img_xray[n]>1)[0])/self.SENSOR_IMAGE_SIZE)  for n in range(self.N_BOTS) ])















    """ Section: Rendering """

    def sample_render_fig(self, local_sensors=True, reward_signal=True, **nullargs):
        # NOTE: this is called by video renderer
        if local_sensors:
            if reward_signal:
                fr_multiplier = 5/3
            else:
                fr_multiplier = 4/3
        else:
            if reward_signal:
                fr_multiplier = 4/3
            else:
                fr_multiplier = 1
        fig = plt.figure(
            layout='constrained',
            figsize=(   (self.X_RANGE)*2*self.render_figure_ratio *fr_multiplier ,
                        (self.Y_RANGE)*2*self.render_figure_ratio ), dpi=self.render_dpi)
        plt.close()
        return fig
    
    #==============================================================
        
    def render(self, local_sensors=True, reward_signal=True, show_plots=True):

        if local_sensors:
            if reward_signal:
                fig = plt.figure(
                    layout='constrained',
                    figsize=(   (self.X_RANGE)*2*self.render_figure_ratio *5/3 ,
                            (self.Y_RANGE)*2*self.render_figure_ratio   ), dpi=self.render_dpi)
                subfigs = fig.subfigures(1, 3, wspace=0.02, width_ratios=[1, 3, 1]) # row, col
                sf_sensor_data, sf_state, sf_reward = subfigs[0], subfigs[1], subfigs[2]
            else:
                fig = plt.figure(
                    layout='constrained',
                    figsize=(   (self.X_RANGE)*2*self.render_figure_ratio *4/3 ,
                            (self.Y_RANGE)*2*self.render_figure_ratio   ), dpi=self.render_dpi)
                subfigs = fig.subfigures(1, 2, wspace=0.02, width_ratios=[1, 3]) # row, col
                sf_sensor_data, sf_state = subfigs[0], subfigs[1]

        else:
            if reward_signal:
                fig = plt.figure(
                    layout='constrained',
                    figsize=(   (self.X_RANGE)*2*self.render_figure_ratio *4/3 ,
                            (self.Y_RANGE)*2*self.render_figure_ratio   ), dpi=self.render_dpi)
                subfigs = fig.subfigures(1, 2, wspace=0.02, width_ratios=[3, 1]) # row, col
                sf_state, sf_reward = subfigs[0], subfigs[1]
            else:
                fig = plt.figure(
                    layout='constrained',
                    figsize=(   (self.X_RANGE)*2*self.render_figure_ratio,
                            (self.Y_RANGE)*2*self.render_figure_ratio   ), dpi=self.render_dpi)
                subfigs = fig.subfigures(1, 1, wspace=0.02, width_ratios=[1,]) # row, col
                sf_state = subfigs


        
        # ============================================================================================================
        # draw sensor data on left
        # ============================================================================================================
        if local_sensors:
            sf_sensor_data.suptitle(f'Sensor Data')
            ax = sf_sensor_data.subplots(self.N_BOTS, 1)
            limL, limH = -self.SCAN_RADIUS*1.25, self.SCAN_RADIUS*1.25
            for n in range(self.N_BOTS):
                ax[n].axis('equal')
                if self.alive[n]:
                    ax[n].set_title(self.names[n])
                    ax[n].set_xlim(limL, limH)
                    ax[n].set_ylim(limL, limH)
                    ax[n].scatter([0], [0], color=self.colors[n], marker = self.markers[n])
                    ax[n].add_patch( # speedometer
                        Circle(   ( 0, 0 ), ( self.BOT_RADIUS), color=self.colors[n], linewidth=2.0, fill=False))
                    ax[n].add_patch( # scan radius
                        Circle(   ( 0, 0 ), ( self.SCAN_RADIUS), color=self.colors[n] if self.alive[n] else 'black', linewidth=0.5, fill=False))
                    ax[n].add_patch( # safe distance
                        Circle(   ( 0, 0 ), ( self.SAFE_CENTER_DISTANCE -self.BOT_RADIUS), color='black', linewidth=2.0, fill=False, linestyle='dotted'))
                
                    neighbours = self.sensor_data [ n, np.where(self.sensor_data[n, :, 0]>0)[0], :]
                    ax[n].set_xlabel(f'Total: {self.alln[n, 0]:.0f}, Occluded: {self.occn[n, 0]:.0f}')
                    for g in neighbours:
                        xp,     yp,     xn,     yn,  x,     y,      dx,     dy,   occ = \
                        g[1],   g[2],   g[3],  g[4], g[5],  g[6],   g[7],   g[8],  g[-1]
                        # the shadow arcs vectors and tangesnts
                        #ax.annotate(str((180/np.pi)*get_angle((x,y))) , xy=(x,y)) #<--- angle in degrees
                        line_style_arc = ('dashed' if occ else 'solid')
                        ax[n].annotate("",  xytext=(0, 0), xy=( 0 + x, 0 + y ),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc , color='black')) # distance vector
                        ax[n].annotate("", xytext=(0, 0), xy=(0 + xn, 0 +yn ),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc, color="tab:red")) # neg arc
                        ax[n].annotate("",  xytext=(0, 0), xy=( 0 + xp, 0 + yp),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_arc, color="tab:green")) # pos arc


                        # vectors - from arch mid to low and high
                        line_style_vec = ('dotted' if occ else 'solid')
                        ax[n].annotate("", xytext=(0, 0), xy=( 0 + dx, 0 + dy ),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color='black'))
                        ax[n].annotate("", xytext=(x, y), xy=(0 + xn, 0 +yn ),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color="tab:red"))
                        ax[n].annotate("", xytext=(x, y), xy=( 0 + xp, 0 + yp),
                            arrowprops=dict(arrowstyle="->", linewidth=0.7, linestyle=line_style_vec, color="tab:green"))
                        ax[n].add_patch( 
                            Circle(   ( dx, dy ), ( self.BOT_RADIUS ),    
                                color=('tab:grey' if occ else 'black') , linewidth=2.0, fill=False, linestyle=line_style_vec))



        # ============================================================================================================
        # draw state in middle
        # ============================================================================================================


        sf_state.suptitle(f'{self.name}/{self.choose_i_state} :: Step-{self.ts}')
        ax = sf_state.subplots(1, 1)
        # bounding box width 
        ax.axis('equal')
        ax.set_xlim((-self.X_RANGE-int(self.X_RANGE*self.render_bounding_width), self.X_RANGE+int(self.X_RANGE*self.render_bounding_width)))
        ax.set_ylim((-self.Y_RANGE-int(self.Y_RANGE*self.render_bounding_width), self.Y_RANGE+int(self.Y_RANGE*self.render_bounding_width)))
        ax.vlines(-self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed')
        ax.vlines(self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(-self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        if self.TARGET_RADIUS>0:
            ax.add_patch( # target circle
                Circle(   ( 0, 0 ), ( self.TARGET_RADIUS ),color='black', linewidth=0.5, fill=False, linestyle='dashed'))

        self.render_state_hook(ax)# call the hook now

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



        # ============================================================================================================
        # draw signals on right
        # ============================================================================================================

        if reward_signal:
            if self.record_reward_hist:
                if self.enable_imaging:
                    bx = sf_reward.subplots(6, 1, gridspec_kw={'height_ratios': [1, 1, 4, 1, 1, 1]})
                    ax_xray, ax_dray, ax_rsig, ax_rsum, ax_srew, ax_crew = bx[0], bx[1], bx[2], bx[3], bx[4], bx[5]
                else:
                    bx = sf_reward.subplots(4, 1, gridspec_kw={'height_ratios': [4, 1, 1, 1]})
                    ax_rsig, ax_rsum, ax_srew, ax_crew  = bx[0], bx[1], bx[2], bx[3]
            else:
                if self.enable_imaging:
                    bx = sf_reward.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 4]})
                    ax_xray, ax_dray, ax_rsig = bx[0], bx[1], bx[2]
                else:
                    bx = sf_reward.subplots(1, 1, gridspec_kw={'height_ratios': [1,]})
                    ax_rsig= bx


            # reward_signal
            ax_rsig.set_xticks(self.reward_plot_x, self.reward_labels)
            if self.render_normalized_reward:
                ax_rsig.set_title('Reward Signal (Normalized)')
                ax_rsig.set_ylim(0.0,1.1)
                ax_rsig.bar(self.reward_plot_x, self.reward_signal/self.rw8, color='tab:green' )
                for i in range(self.r_dim):
                    ax_rsig.annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.125, 1.05)  )
                    #ax_rsig.vlines(i, 0, self.rw8[i], color='black', linestyle='dotted', linewidth=0.5)
            else:
                ax_rsig.set_title('Reward Signal')
                ax_rsig.set_ylim(0.0,self.reward_plot_y+0.1)
                ax_rsig.bar(self.reward_plot_x, self.reward_signal, color='tab:green' )
                for i in range(self.r_dim):
                    ax_rsig.annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.125, self.reward_signal[i]+0.05)  )
                    ax_rsig.vlines(i, 0, self.rw8[i], color='black', linestyle='dotted', linewidth=0.5)



            if self.record_reward_hist:
                # reward plots
                ax_srew.plot(self.reward_hist[1], color='tab:blue')
                ax_srew.set_title(f'Step-Reward : {self.reward_hist[1][-1]:.3f}')

                ax_crew.plot(self.reward_hist[2], color='tab:brown')
                ax_crew.set_title(f'Cummulative-Reward : {self.reward_hist[2][-1]:.3f}')
                
                ax_rsum.plot(self.reward_hist[0], color='tab:green')
                ax_rsum.set_title(f'Signal Sum : {self.reward_hist[0][-1]:.3f}')

            if self.enable_imaging:
                ax_xray.set_yticks(range(self.N_BOTS), self.names)
                ax_xray.set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
                ax_xray.grid(axis='both')
                ax_xray.imshow(self.img_xray, aspect=self.img_aspect, cmap= self.render_xray_cmap, vmin= 0, vmax= self.N_BOTS )
                ax_xray.set_title("X-Ray: All Sensors")
                
                ax_dray.set_yticks(range(self.N_BOTS), self.names)
                ax_dray.set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
                ax_dray.grid(axis='both')
                ax_dray.imshow(self.img_dray, aspect=self.img_aspect, cmap= self.render_dray_cmap, vmin= 0, vmax= self.SCAN_RADIUS )
                ax_dray.set_title("D-Ray: All Sensors")
        

        
        (plt.show() if show_plots else plt.close())
        return fig

    #==============================================================
    def render_state_hook(self, ax):
        pass # <---- use 'ax' to render target points

    def get_render_handler(self, 
            render_mode,  # str 'all', 'env', 'rew', 'sen'
            save_fig, # str - name of folder in ehich to save rendered plots or name of video if make_video is true (auto append .avi) 
            save_dpi, # 'figure' or a value for dpi - this is passed to fig.savefig() and overwrites render_dpi 
            make_video, # bool - if True, makes a video of all rendered frames
            video_fps,
            ):
        return RenderHandler(self, render_mode=render_mode, save_fig=save_fig, save_dpi=save_dpi, make_video=make_video, video_fps=video_fps)



class RenderHandler:

    render_modes = { #<-- define only when rendering
        'all':      lambda s :dict(local_sensors=True,  reward_signal=True, show_plots=s, ),
        'env':      lambda s :dict(local_sensors=False, reward_signal=False, show_plots=s, ),
        'rew':      lambda s :dict(local_sensors=False, reward_signal=True, show_plots=s, ),
        'sen':      lambda s :dict(local_sensors=True,  reward_signal=False, show_plots=s, ),
        }
    
    def __init__(self, env, render_mode, save_fig, save_dpi, make_video, video_fps) -> None:
        self.env=env
        self.render_mode = render_mode
        self.save_fig=save_fig
        self.save_dpi = save_dpi
        self.make_video=make_video
        self.video_fps=video_fps

        # make render functions
        self.Start = self.noop
        self.Render = self.noop
        self.Stop = self.noop
        if render_mode:
            self.render_kwargs = self.render_modes[self.render_mode](not(save_fig))
            if save_fig:
                if make_video:
                    self.Start = self.Start_Video
                    self.Render = self.Render_Video
                    self.Stop = self.Stop_Video
                else:
                    self.Start = self.Start_Image
                    self.Render = self.Render_Image
            else:
                self.Render = self.Render_Show

    def noop(self):
        pass

    def Render_Show(self):
        self.env.render(**self.render_kwargs)

    def Start_Image(self):
        os.makedirs(self.save_fig, exist_ok=True)
        self.n=0
    
    def Render_Image(self):
        self.env.render(**self.render_kwargs).savefig(os.path.join(self.save_fig, f'{self.n}.png'), 
                        dpi=self.save_dpi, transparent=False )     
        self.n+=1

    def Start_Video(self):
        # video handler requires 1 env.render to get the shape of env-render figure
        self.buffer = BytesIO()

        print(f'[{__class__.__name__}]:: Reseting environment for first render...')
        #self.env.reset()

        #self.buffer.seek(0) # seek zero before writing - not required on first write
        self.env.sample_render_fig(**self.render_kwargs).savefig( self.buffer, dpi=self.save_dpi, transparent=False ) 
        self.buffer.seek(0) # seek zero before reading
        frame = cv2.imdecode(np.asarray(bytearray(self.buffer.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.height, self.width, _ = frame.shape
        self.video_file_name = self.save_fig+'.avi'

        #                                   file,     fourcc, fps, size
        self.video = cv2.VideoWriter(self.video_file_name , 0, self.video_fps, (self.width, self.height)) 

        # self.video.write(frame) #<--- do not write yet
        print(f'[{__class__.__name__}]:: Started Video @ [{self.video_file_name}] :: Size [{self.width} x {self.height}]')

    def Render_Video(self):
        self.buffer.seek(0) # seek zero before writing 
        self.env.render(**self.render_kwargs).savefig( self.buffer, dpi=self.save_dpi, transparent=False ) 
        self.buffer.seek(0) # seek zero before reading
        self.video.write(cv2.imdecode(np.asarray(bytearray(self.buffer.read()), dtype=np.uint8), cv2.IMREAD_COLOR))

    def Stop_Video(self):
        cv2.destroyAllWindows()
        self.video.release()
        self.buffer.close()
        del self.buffer
        print(f'[{__class__.__name__}]:: Stopped Video @ [{self.video_file_name}]')


""" ARCHIVE


    def render_sensor_image(self, n, use_xray=False, show_ticks=True):
        if not self.enable_imaging:
            return None, ""
        fig,_ = plt.figure()
        if show_ticks:
            plt.xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
            
            plt.grid(axis='both')

        if use_xray:
            plt.imshow(np.reshape( (self.img_xray[n, :]), (1, self.SENSOR_IMAGE_SIZE) ), aspect=self.img_aspect, 
                            cmap= self.render_xray_cmap, vmin= 0, vmax= self.N_BOTS ) 
            arcm = np.where(self.img_oray[n]>0)[0]
            for arcpt in arcm:
                plt.scatter( [arcpt], [0], color='white', marker='d') # self.img_oray[n,arcpt:arcpt+1]
                # f'${self.img_oray[n,arcpt]}$'
                plt.annotate(f'{self.img_oray[n,arcpt]}', xy=(arcpt,0.4))
            plt.title("X-Ray: "+self.names[n])
        else:
            plt.imshow(np.reshape( (self.img_dray[n, :]), (1, self.SENSOR_IMAGE_SIZE) ), aspect=self.img_aspect, 
                            cmap= self.render_dray_cmap, vmin= 0, vmax= self.SCAN_RADIUS )
            plt.title("D-Ray: "+self.names[n])
        
        
        (plt.show() if self.show_plots else plt.close())
        return fig, "render_sensor_image_local"

"""