#==============================================================
import numpy as np
from math import ceil, inf, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import gym, gym.spaces
#import itertools
from scipy.spatial import ConvexHull   #, convex_hull_plot_2d
from .common import REMAP, get_angle, get_nspace, get_espace
import torch as tt



class SwarmState:
    # describes a swarm initial state
    def __init__(self,
        target_point=None, # keep None for random target
        target_radius=0.0, # optional
        reset_noise=0.0, # reset noise in x-y direction
        points=None, # the coordinates of robots
        **kwargs):
        self.target_point=target_point
        self.target_radius=target_radius
        self.reset_noise=reset_noise
        assert points is not None, f'Initial coordinates missing!'
        self.points=points
        for k,v in kwargs.items(): setattr(self, k, v)
    
    @staticmethod
    def points_from_img(path, x_range, y_range):
        # expects an image, extract the coordinates of black points
        try:
            import cv2 # pip install opencv-python
        except:
            print(f'[!] failed to import cv2!')
            return None
        img= cv2.imread(path, 0)
        scale = ((x_range*2)/img.shape[0], (y_range*2)/img.shape[1])
        translate = (-x_range, -y_range)
        print(f'[GEN]@ {path}\n[AUTO-SCALE: {scale}]\n[AUTO-TRANSLATE: {translate}]')
        X,Y = np.where(img==0)
        points = [(scale[0]*float(y) + translate[0],  scale[1]*float(img.shape[1]-x) + translate[1]) for x,y in zip(X,Y)]
        return points

class World(gym.Env):

    r"""
    Update:

    target_point, target_radius, reset_noise are now part of initial state
    random_target removed, is now part of initial state
    state_target is always True
    
    """

    default_bot_colors = [ 'red', 'blue', 'green', 'gold',   
                          'cyan', 'magenta', 'purple', 'brown', 
                          'tab:orange', 'tab:pink', 'tab:green', 'tab:blue',
                           'tab:olive', 'tab:red', 'tab:brown', 'coral',
                            'lime', 'slategrey', 'tab:cyan', 'tab:purple' ]
    default_n_bots = len(default_bot_colors)
    default_bot_markers = ['o' for _ in range(default_n_bots)]

    # Data-types
    STATE_DTYPE =    np.float32
    ACTION_DTYPE =   np.float32
    REWARD_DTYPE =   np.float32
    
    def info(self):
        return \
            f'{self.name} :: Dim: ( X={self.X_RANGE*2}, Y={self.Y_RANGE*2}, H={self.horizon} )' + \
            f'\nDelta-Reward: [{self.delta_reward}],  Delta-Action: [{self.delta_action_mode}]' + \
            f'\nImaging: [{self.enable_imaging}],  History: [{self.record_reward_hist}]\n'

    def add_initial_states(self, *states):
        for state in states:
            assert len(state.points)==self.N_BOTS, f'state must have exactly {self.N_BOTS} point'
            self.initial_states.append(state)
        return self 

    def __init__(self, name='default', x_range=10, y_range=10, 
                enable_imaging=True, horizon=0, seed=None, reward_scheme=None, delta_reward=False,
                    n_bots=4,             # number of bots in swarm
                    bot_radius=1.0,         # meters - fat-bot body radius
                    scan_radius=15.0,        # scannig radius of onboard sensor (neighbourhood)
                    safe_distance=3.0,      # min edge-to-edge distance b/w bots (center-to-center distance will be +(2*BOT_RADIUS))
                    speed_limit=1.0,        # [+/-] upper speed (roll, pitch) limit of all robots
                    delta_speed=0.0,        # [+/-] constant linear acceleration (throttle-break) - keep 0 for direct action mode
                    sensor_resolution=40,  # choose based on scan distance, use form: n/pi , pixel per m
                    min_bots_alive=0,     # min no. bots alive, below which world will terminate # keep zero for all alive condition
                    record_state_history=0,
                    record_reward_hist=True, render_normalized_reward=True,
                    render_xray_cmap='hot', render_dray_cmap='copper',  render_dpi=None,
                    render_figure_ratio=0.4, render_bounding_width=0.05) -> None:
        super().__init__()
        self.name = name
        self.X_RANGE, self.Y_RANGE = float(x_range), float(y_range)

        self.POINT_CLIP_LOW = np.array([-self.X_RANGE, -self.Y_RANGE])
        self.POINT_CLIP_HIGH = np.array([self.X_RANGE, self.Y_RANGE])
        self.N_BOTS = int(n_bots)
        self.nr = range(self.N_BOTS)
        self.BOT_RADIUS = float(bot_radius)
        self.SCAN_RADIUS = float(scan_radius)
        self.SAFE_CENTER_DISTANCE = float(safe_distance) + self.BOT_RADIUS * 2
        self.SPEED_LIMIT = float(speed_limit)
        self.DELTA_SPEED = float(delta_speed)
        self.delta_action_mode = (self.DELTA_SPEED!=0.0)
        self.SENSOR_RESOULTION = float(sensor_resolution)/pi
        self.SENSOR_UNIT_ARC = self.SENSOR_RESOULTION * self.SCAN_RADIUS
        self.SENSOR_IMAGE_SIZE = int(ceil(2 * np.pi * self.SENSOR_UNIT_ARC)) # total pixels in sensor data input
        self.MIN_BOTS_ALIVE = (min_bots_alive if min_bots_alive>0 else self.N_BOTS)
        self.enable_imaging = enable_imaging
        self.set_horizon(horizon)
        self.record_reward_hist = record_reward_hist
        self.render_normalized_reward = render_normalized_reward

        assert reward_scheme is not None, f'reward_scheme not defined'
        assert len(reward_scheme)>0, f'reward_scheme requires at least one signal'
        self.reward_scheme = reward_scheme # a dict of reward signal (RF_* : weight)
        self.clean_reward_scheme()
        self.delta_reward = delta_reward
        self.rng = np.random.default_rng(seed)
        self.enable_state_hist = (record_state_history>0)
        self.record_state_history=record_state_history
        
        self.initial_states=[]

        # for rendering
        self.render_xray_cmap = render_xray_cmap
        self.render_dray_cmap = render_dray_cmap
        self.render_figure_ratio = render_figure_ratio
        self.render_bounding_width = render_bounding_width
        self.render_dpi = render_dpi
        self.MAX_SPEED = sqrt(2) * self.SPEED_LIMIT # note - this is magnitude of MAX velocity vector
        self.colors = self.default_bot_colors[0:self.N_BOTS]
        self.markers = self.default_bot_markers[0:self.N_BOTS]
        self.names = self.default_bot_colors[0:self.N_BOTS]
        self.img_aspect = self.SENSOR_IMAGE_SIZE/self.SENSOR_RESOULTION # for rendering
        self.arcDivs = 4 + 1  # no of divisions on sensor image
        self.arcTicks = np.array([ (int( i * self.SENSOR_UNIT_ARC), round(i*180/pi, 2)) \
                                    for i in np.linspace(0, 2*np.pi, self.arcDivs) ])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.build() # call default build
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # counter
        self.episode = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # render related
        print(f'[*] World Created :: {self.info()}')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_horizon(self, horizon): self.horizon = (horizon if horizon>0 else inf)
    def clean_reward_scheme(self):
        zero_keys = []
        for k,v in self.reward_scheme.items():
            if v==0: zero_keys.append(k)
        for k in zero_keys: 
            print(f'[!] removing zero-weight reward [{k}]')
            del self.reward_scheme[k]
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Section: Build """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

    def build(self):
        # --> call in order
        self.build_observation_space()
        self.build_action_space()
        self.build_vectors()
        self.build_reward_signal()
        
    def build_observation_space(self):
        # now define observation_space ----------------------------------------------------------------------------
        position_space_info = {  # Position/Direction x,y
                        'dim': 2,
                        'low': (-self.X_RANGE, -self.Y_RANGE), 
                        'high':( self.X_RANGE, self.Y_RANGE), 
                    }
        self.force_clip_low = np.array((-self.X_RANGE, -self.Y_RANGE), dtype= self.STATE_DTYPE)
        self.force_clip_high= np.array((self.X_RANGE, self.Y_RANGE), dtype= self.STATE_DTYPE)
        velocity_space_info = { # velocities dx, dy
                        'dim': 2,
                        'low': (-self.SPEED_LIMIT, -self.SPEED_LIMIT), 
                        'high':( self.SPEED_LIMIT,  self.SPEED_LIMIT), 
                    }
        neighbour_space_info = { # total_neighbours, # occluded neighbours
                        'dim': 2,  
                        'low': (0,             0,         ), 
                        'high':(self.N_BOTS-1, self.N_BOTS-2), 
                    }

        sensor_space_info = { # ( dis, arch, arcl, arcm ) * n_bots-1
                        'dim': (self.N_BOTS-1) * 4,  
                        'low': tuple(np.array([(0, 0, 0, 0) for _ in range(self.N_BOTS-1)]).flatten()), 
                        'high': tuple(np.array([(max(self.X_RANGE, self.Y_RANGE), 2*pi, 2*pi, 2*pi) for _ in range(self.N_BOTS-1)]).flatten()), 
                    }
        # if False:
        #     self.observation_space = get_nspace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
        #                 shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'] + sensor_space_info['dim'], ),
        #                 low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low'] + sensor_space_info['low']  ),
        #                 high=   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] + sensor_space_info['high'] ))
            
        #     self.o_dim = 6 + ((self.N_BOTS-1) * 4)
        # else:
        if self.delta_action_mode:
            shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'] , )
            low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low']   )
            high =   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] )
            odim = 6
        else: # dont include velocities in observation space
            shape = (position_space_info['dim']   +  neighbour_space_info['dim'] , )
            low=    (position_space_info['low']   +  neighbour_space_info['low']   )
            high =   (position_space_info['high']  + neighbour_space_info['high'] )
            odim = 4


        self.e_dim = 3
        self.observation_space = get_espace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
            shape = shape,
            edim = self.e_dim, # for x and y target point and ratget radius
            low = low,
            high = high,
            elow = (-self.X_RANGE, -self.Y_RANGE, 0),
            ehigh = (self.X_RANGE, self.Y_RANGE, max(self.X_RANGE, self.Y_RANGE))
            )
        self.o_dim = odim 
        self.base_observation = np.zeros(self.observation_space.shape, self.observation_space.dtype)
        self.target_observation = self.base_observation[-self.e_dim:]
        self.observation = self.base_observation[0:-self.e_dim].reshape((self.N_BOTS, self.o_dim))
        self.initial_observation = np.zeros_like(self.observation)

        # define observation views
        self.xy =           self.observation [:, 0:2] # x,y
        self.x =            self.observation [:, 0:1] # x
        self.y =            self.observation [:, 1:2] # y

        if self.delta_action_mode:
            self.dxy =           self.observation [:, 2:4] # dx,dy
            self.dx =            self.observation [:, 2:3] # dx
            self.dy =            self.observation [:, 3:4] # dy

            self.alln =            self.observation[:, 4:5]
            self.occn =            self.observation[:, 5:6]
        else:
            self.dxy =           np.zeros_like(self.xy) # dx, dy
            self.dx =            self.dxy [:, 0:1] # dx
            self.dy =            self.dxy [:, 1:2] # dy

            self.alln =            self.observation[:, 2:3]
            self.occn =            self.observation[:, 3:4]

        #self.sense =            self.observation[:, 6:].reshape((self.N_BOTS, (self.N_BOTS-1) , 4))
        self.sense =            np.zeros((self.N_BOTS, (self.N_BOTS-1) , 4), dtype=self.STATE_DTYPE)
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
        
        self.dsafe=np.zeros_like(self.dmat) + self.SAFE_CENTER_DISTANCE
        for i in self.nr:
            self.dsafe[i,i]=0


        # sensor images
        if self.enable_imaging:
            self.img_oray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)         
            self.img_xray = np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16) 
            self.img_dray =  np.zeros((self.N_BOTS, self.SENSOR_IMAGE_SIZE ), dtype=np.int16)
        
        self.state_hist = []
        return


    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Section: State Dynamics """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

    def kill_bots(self, *bots):
        for b in bots:
            self.alive[b] = False
            self.dmat[b,:]=0
            self.dmat[:,b]=0
            self.sensor_data[b, :, :] = 0 
            self.sense[b, :, :] = 0
            if self.enable_imaging:
                self.img_xray[b,:]=0
                self.img_oray[b,:]=0
                self.img_dray[b,:]=self.SCAN_RADIUS
        self.n_alive = np.sum(self.alive)

    def update_distances(self):
        for f in self.nr:
            if self.alive[f]:
                #self.dmat[f,f]=0.0
                #if not self.force_field:
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
        for b in self.nr:
            
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
            self.sense[b, :, :] = 0
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

                ahigh = get_angle( ( xpos, ypos   ) )
                amid = get_angle( ( x,    y      ) ) 
                alow = get_angle( ( xneg, yneg   ) ) 

                arch = int(ahigh * self.SENSOR_UNIT_ARC)
                arcm = int(amid * self.SENSOR_UNIT_ARC)
                arcl = int(alow * self.SENSOR_UNIT_ARC)

                self.sensor_data[b, i, : ] = (1, xpos, ypos, xneg, yneg,  
                                                x, y, difx, dify, distance, 
                                                ahigh, alow, amid, 0)
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
                
            # ---> all neighbours end
            #distance_key = 9
            high_key, low_key = 10, 11
            self.occn[b,:] = 0
            self.alln[b,:] = nos_N

            for ni in range(1, nos_N):
                #print(f'neig: {ni}')
                
                # the nearest bot is already sorted # if lower bots are occulded or not
                zi = ni-1
                while zi>=0:
                    ah, al = self.sensor_data[b, ni, high_key], self.sensor_data[b, ni, low_key]
                    zh, zl = self.sensor_data[b, zi, high_key], self.sensor_data[b, zi, low_key]

                   # if zh<zl:
                        #zl -= 2*pi
                    #print(f'[1] {al=}, {ah=}, {zl=}, {zh=}')
                    
                    # Q> is A(al, ah) occluded by Z(zl, zh)
                    is_occluded = False
                    ###------------four cases:
                    a_oT = (al>ah)
                    z_oT = (zl>zh)
                    if z_oT:
                        if a_oT:
                            is_occluded=True
                        else:
                            is_occluded = (zh>al or ah>zl)
                            # zh al zl ah - occ-part-low
                            # al zh ah zl - occ-part-high
                            # al ah zh zl - full occ
                            # zh zl al ah - full occ
                    else:
                        if a_oT:
                            is_occluded = (ah>zl or zh>al)
                            # zl ah zh al - occ-part-low
                            # ah zl al zh - occ-part-high
                        else:
                            if al<zl:
                                is_occluded = (ah>zl)
                            else:
                                is_occluded = (al<zh)
                            # al zl ah zh - occ-part-low
                            # zl al zh ah - occ-part-high
                            # zl al ah zh - occ-full
                        



                        
                        
                    #print(f'[2] {al=}, {ah=}, {zl=}, {zh=}')
                    #if al>zl: #first determine lower
                    #    zl, zh, al, ah = al, ah, zl, zh
                    #print(f'[3] {al=}, {ah=}, {zl=}, {zh=}')
                    #print( zl, zh, al, ah)
                    #if not (al<zl and ah<zh and ah<zl):
                    if is_occluded:
                        self.sensor_data[b, ni, -1] = 1
                        self.occn[b,:]+=1
                        zi=-1
                    else:
                        zi-=1


                    self.sense[b, ni, :] = self.sensor_data[b, ni, 9:13 ]


                
            
        # ------------------> end for all bots
        
        
        return



    def is_timeout(self):
        return bool( ( self.ts>=self.horizon ) )

    def is_terminal(self):
        return bool( ( self.n_alive < self.MIN_BOTS_ALIVE ) )
    
    def save_state(self, path):
        state = SwarmState(
            target_point=tuple(self.TARGET_POINT),
            target_radius=float(self.TARGET_RADIUS),
            reset_noise=self.initial_state.reset_noise,
            points=[ tuple(p) for p in self.observation[:, 0:2] ],
            terminal= int(self.terminal),
            timeout= int(self.timeout),
            steps=self.ts,
            )
        tt.save(state, path)




    def reset(self, starting_state=None):

        self.episode += 1
        
        if starting_state is None:
            # reset - choose randomly from known initial state distribution 
            self.initial_state = self.initial_states[self.rng.integers(0, len(self.initial_states))]
        else:
            self.initial_state = starting_state

        for i,p in  enumerate(self.initial_state.points):
            self.initial_observation[i, 0:2] = np.array(p[0:2]) + \
                ( self.rng.uniform(-self.initial_state.reset_noise, self.initial_state.reset_noise) \
                 if self.initial_state.reset_noise!=0 else 0.0)

        if self.initial_state.target_point is None:
            rand_point = np.array( [
                        self.rng.uniform(-self.X_RANGE*.8, self.X_RANGE*.8),
                        self.rng.uniform(-self.Y_RANGE*.8, self.Y_RANGE*.8),
                        ] )

        else:
            rand_point=np.array(self.initial_state.target_point)
             

        self.TARGET_RADIUS = self.initial_state.target_radius
        self.TARGET_POINT = rand_point
        #np.clip(rand_point, self.POINT_CLIP_LOW, self.POINT_CLIP_HIGH)
        
        self.observation[:] = self.initial_observation # copy state vectors
        self.target_observation[0:2] = self.TARGET_POINT
        self.target_observation[2] = self.TARGET_RADIUS 
        self.alive[:]=True # all robots alive flag
        self.n_alive = np.sum(self.alive)
        if self.enable_state_hist: 
            self.state_hist.clear()
            #self.state_hist_count=0
            self.state_hist.append(np.copy(self.xy))
            self.state_hist_count=1
        self.actuator*=0 # reset actuator
        # update state variables----------------------
        self.update_distances()
        self.update_sensor_data()
        #---------------------------------------------

        self.reward_signal[:] = self.get_reward_signal()
        self.reward_signal_sum = np.sum(self.reward_signal)
        self.step_reward = 0.0
        self.cummulative_reward = 0.0
        self.ts=0
        self.terminal = self.is_terminal()
        self.timeout = self.is_timeout()
        self.done = self.terminal or self.timeout
        if self.record_reward_hist:
            self.reward_hist = [ [],[],[]
                #[self.reward_signal_sum ], 
                #[self.step_reward], 
                #[self.cummulative_reward]
                ] # signal sum, current reward, cumm reward
        
        return self.base_observation
    

    def read_actuators(self, action):
        self.base_actuator[:] = self.action_mapper.in2map(np.clip( action, self.action_space.low, self.action_space.high )) # copy actions
        self.actuator[np.where(self.alive==False)[0],:]=0 # not act on crahsed robots

    def take_action(self):
        # set actuator values
        if self.delta_action_mode:
            self.dxy[:] = np.clip(self.dxy+self.δxy, -self.SPEED_LIMIT, self.SPEED_LIMIT)
        else:
            self.dxy[:] = self.δxy
        # assume that veclocities are updates - x,y,, dx,dy
        #if self.force_field:
            #self.xy[:]=np.clip(self.dxy+self.xy, self.force_clip_low, self.force_clip_high  ) # move forward - restricted
        #else:
        self.xy+=self.dxy # move forward - unrestricted
        if self.enable_state_hist:  
            self.state_hist.append(np.copy(self.xy))
            if self.state_hist_count>self.record_state_history:
                del self.state_hist[0]
            else:
                self.state_hist_count+=1
            
            
        # update state variables----------------------
        self.update_distances()
        self.update_sensor_data()
        #---------------------------------------------

        self.ts+=1
        self.terminal = self.is_terminal()
        self.timeout = self.is_timeout()
        self.done = self.terminal or self.timeout

        #**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--
        if not self.done:
            # reward is based on wighted signal
            current_reward_signal = self.get_reward_signal()
            current_reward_signal_sum = np.sum(current_reward_signal)
            # in delta mode, if current reward_signal is higher, then give +ve reward
        
            self.step_reward = float( 
                (current_reward_signal_sum - self.reward_signal_sum) \
                    if self.delta_reward else \
                current_reward_signal_sum ) 


            self.reward_signal[:] = current_reward_signal
            self.reward_signal_sum = current_reward_signal_sum
        else:
            self.step_reward=-1.0
        #**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--**--

        self.cummulative_reward += self.step_reward
        if self.record_reward_hist:
            self.reward_hist[0].append(self.reward_signal_sum)
            self.reward_hist[1].append(self.step_reward)
            self.reward_hist[2].append(self.cummulative_reward)
        return self.base_observation, self.step_reward, self.done, {}

    def step(self, action):
        self.read_actuators(action)
        return self.take_action()

    def step_random(self):
        self.step(self.action_space.sample())
    
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Section: Reward Signal : implement in inherited classes """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

    @staticmethod
    def get_default_reward_signal(
        dis_target_point=   1.0, 
        dis_neighbour =     1.0,
        hull_formed=        1.0,
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
    ): # this is created for autocomplete
        res = {}
        if dis_target_point!=0:     res['dis_target_point']=dis_target_point
        if dis_neighbour!=0:        res['dis_neighbour']=dis_neighbour
        if hull_formed!=0:          res['hull_formed']=hull_formed
        if dis_target_radius!=0:    res['dis_target_radius']=dis_target_radius
        if all_unsafe!=0:           res['all_unsafe']=all_unsafe
        if all_neighbour!=0:        res['all_neighbour']=all_neighbour
        if occluded_neighbour!=0:   res['occluded_neighbour']=occluded_neighbour
        return res

    def get_default_reward_data(self):
        max_dis_bots = ((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS
        max_n_bots = (self.N_BOTS-1)*self.N_BOTS
        #max_o_bots = (2*self.N_BOTS-3)*self.N_BOTS
        max_n_unsafe = (self.SAFE_CENTER_DISTANCE**self._scd_power)*self.N_BOTS
        maxd = (self.X_RANGE**2+self.Y_RANGE**2)**0.5
        max_g_bots = maxd*self.N_BOTS*(self.N_BOTS-1) * 0.5
        return dict(
                                    #  sign,      low,      high              label

            # distance to target point <--- lower is better
            dis_target_point =    (    -1,         0,       max_dis_bots,     'C2P-Target',     ),

            # distance to target radius <--- lower is better
            dis_target_radius =   (    -1,         0,       max_dis_bots,     'C2R-Target',     ),

            # no of unsafe bots  <--- lower is better
            all_unsafe =          (    -1,         0,       max_n_unsafe,       'Safe-Bots',    ),

            # no of total neighbours  <--- higher is better
            all_neighbour =       (     1,         0,       max_n_bots,       'Neighbours'    ),

            # no of occluded neighbours  <--- lower is better
            occluded_neighbour = (    -1,         0,       max_n_bots,       'V-Neighbours'  ),

            # distance b/w neighbours
            dis_neighbour =      (    -1,         0,       max_g_bots,       'D-Neighbours'  ),

            # occlusion ratio = occluded pixels / total pixels  <--- lower is better
            # occlusion_ratio =     (    -1,         0,       1,                'V-Ratio'       ),

            # convex hull formed or not <---- higher is better
            hull_formed =       (    1,        0,        1,                 'C-Hull'        ),

        )

    def build_reward_signal(self):

        self.reward_data = self.get_default_reward_data()

        reward_labels, rsign, rlow, rhigh, rw8, reward_caller   = [], [], [], [], [], []
        for k,w in self.reward_scheme.items():
            if k in self.reward_data:    
                v = self.reward_data[k]
                rsign.append( v[0] )
                rlow.append( v[1] )
                rhigh.append( v[2] )
                reward_labels.append( v[3] )
                rw8.append(w)
                reward_caller.append( getattr(self, f'RS_{k}') )
            else:
                print(f'[!] Reward Signal [{k}] not found in reward data, skipping...')

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

    def get_multi_reward_signal(self):  # higher is better
        
        return np.array([self.rw8 * self.reward_mapper.in2map(np.array( [ RF(b) for RF in self.reward_caller ], dtype=self.REWARD_DTYPE )) \
            for b in self.nr ], dtype=self.REWARD_DTYPE)


    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Pre-defined reward signals 
    
    Reward signal naming convention:

    Functions for calculating rewards must begin with 'RS_'

        RS_<name>(self, b): -> float
    
        where, b = the bot index (0 to self.N_BOTS-1)

    reward_scheme argument is a dictionary like:

        { <name> : weight }


    """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-


    def RS_dis_neighbour(self, b=None): 
        if b is None: 
            return np.sum(self.dmat)/2
        else:
            return np.sum(self.dmat[b,:])/2

    def RS_dis_target_point(self, b=None): 
        if b is None:
            return np.sum ([ np.linalg.norm( self.xy[n, :]-self.TARGET_POINT, 2 )  for n in self.nr ])
        else:
            return np.linalg.norm( self.xy[b, :]-self.TARGET_POINT, 2 ) 

    def RS_dis_target_radius(self, b=None): 
        if b is None:
            return np.sum ([ np.abs(self.TARGET_RADIUS-np.linalg.norm( self.xy[n, :]-self.TARGET_POINT, 2 ))  for n in self.nr ])
        else:
            return np.abs(self.TARGET_RADIUS-np.linalg.norm( self.xy[b, :]-self.TARGET_POINT, 2 ))

    _scd_power = 0.5
    def RS_all_unsafe(self, b=None):
        if b is None:
            diff = self.dmat-self.dsafe
            du=self.dmat[np.where(diff<0)]
            return np.sum((self.SAFE_CENTER_DISTANCE-du)**self._scd_power)
        else:
            diff = self.dmat[b]-self.dsafe[b]
            du=self.dmat[np.where(diff<0)]
            return np.sum((self.SAFE_CENTER_DISTANCE-du)**self._scd_power)

    def RS_all_neighbour(self, b=None):
        if b is None:
            return np.sum(self.alln)
        else:
            return self.alln[b]

    def RS_occluded_neighbour(self, b=None):
        if b is None:
            return np.sum(self.N_BOTS - 1 - self.alln) + np.sum(self.occn)
        else:
            return (self.N_BOTS - 1 - self.alln[b]) + (self.occn[b])

    def RS_hull_formed(self, b=None):
        self.hull = ConvexHull(self.xy)
        self.hull_formed=(len(self.hull.vertices) == self.N_BOTS)
        return int(self.hull_formed)
    #def RS_occlusion_ratio(self, b=None):
    #    if b is None:
    #        return np.sum ([  (len(np.where(self.img_xray[n]>1)[0])/self.SENSOR_IMAGE_SIZE)  for n in self.nr ])
    #    else:
    #        return (len(np.where(self.img_xray[b]>1)[0])/self.SENSOR_IMAGE_SIZE)
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-
    """ Section: Rendering """
    # $-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-

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
    def render(self, local_sensors=True, reward_signal=True, fill_bots=False, state_hist_marker='o'):

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
            for n in self.nr:
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

        sf_state.suptitle(f'{self.name} :: Step-{self.ts}')
        ax = sf_state.subplots(1, 1)
        # bounding box width 
        ax.axis('equal')
        ax.set_xlim((-self.X_RANGE-int(self.X_RANGE*self.render_bounding_width), self.X_RANGE+int(self.X_RANGE*self.render_bounding_width)))
        ax.set_ylim((-self.Y_RANGE-int(self.Y_RANGE*self.render_bounding_width), self.Y_RANGE+int(self.Y_RANGE*self.render_bounding_width)))
        ax.vlines(-self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed')
        ax.vlines(self.X_RANGE,  -self.Y_RANGE, self.Y_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(-self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        ax.hlines(self.Y_RANGE,  -self.X_RANGE, self.X_RANGE,  color='black', linewidth=0.5, linestyle='dashed'  )
        # axis
        #ax.vlines(0, -self.Y_RANGE, self.Y_RANGE, color='black', linewidth=0.5, linestyle='dashed' )
        #ax.hlines(0, -self.X_RANGE, self.X_RANGE, color='black', linewidth=0.5, linestyle='dashed')
        ax.vlines(self.TARGET_POINT[0], -self.Y_RANGE, self.Y_RANGE, color='black', linewidth=0.5, linestyle='solid' )
        ax.hlines(self.TARGET_POINT[1], -self.X_RANGE, self.X_RANGE, color='black', linewidth=0.5, linestyle='solid')

        if self.TARGET_RADIUS>0:
            ax.add_patch( # target circle
                Circle(   ( self.TARGET_POINT[0], self.TARGET_POINT[1] ), ( self.TARGET_RADIUS ),color='black', linewidth=0.5, fill=False, linestyle='dashed'))

        self.render_state_hook(ax)# call the hook now

        for n in self.nr:
        #--------------------------------------------------------------------------------------------------------------
            botx, boty = self.x[n, 0], self.y[n, 0]
            botcolor = self.colors[n]
            botmarker = self.markers[n]
            bot_speed = np.linalg.norm(self.dxy[n], 2)
            #--------------------------------------------------------------------------------------------------------------

            #--------------------------------------------------------------------------------------------------------------
            if self.enable_state_hist and len(state_hist_marker)>0: 
                for ostate in self.state_hist:
                    ax.scatter(ostate[n,0], ostate[n,1], marker=state_hist_marker, color=botcolor)
                    
            ax.scatter( [botx], [boty], color=botcolor, marker=botmarker )
            if  not self.alive[n]:
                botcolor='tab:gray'
            ax.add_patch( # body (bot radius)
                Circle(   ( botx, boty ), ( self.BOT_RADIUS ),    # 2-tuple center, # float radius
                    color=botcolor, linewidth=1.0, fill=fill_bots, linestyle='solid'))
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

                # self.sensor_data[b, i, : ] = ( 0,1,2,3,4   1, xpos, ypos, xneg, yneg,  
                #                                5,6,7,8,9   x, y, difx, dify, distance, 
                #                                 ahigh, alow, amid, 0)
                for g in neighbours:
                    xp,     yp,     xn,     yn,  x,     y     , lnx, lny = \
                    g[1],   g[2],   g[3],  g[4], g[5],  g[6], g[7], g[8]
                    if fill_bots:
                        tx, ty = botx +lnx, boty +lny
                    else:
                        tx, ty = botx + x, boty + y

                    isocc = bool(g[-1])
                    ax.annotate("", 
                        #xytext="+",
                        xytext=(botx, boty), xy=( tx, ty ),
                        arrowprops=dict(arrowstyle="->", linewidth=0.8, linestyle='dotted', color=('red' if isocc else 'grey')))

                    # ax.annotate("", 
                    #     #xytext="+",
                    #     xytext=(botx, boty), xy=(botx + xn, boty +yn ),
                    #     arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
                    # ax.annotate("", 
                    #     #xytext="+",
                    #     xytext=(botx, boty), xy=( botx + xp, boty + yp),
                    #     arrowprops=dict(arrowstyle="->", linewidth=0.4, linestyle='solid', color=botcolor))
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
                    #ax_rsig.annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.125, self.reward_signal[i]+0.05)  )
                    ax_rsig.annotate(  f'{self.reward_signal[i]:.3f}', xy=(i-0.125, 1)  )
                    ax_rsig.vlines(i, 0, self.rw8[i], color='black', linestyle='dotted', linewidth=0.5)



            if self.record_reward_hist:
                # reward plots
                ax_srew.plot(self.reward_hist[1], color='tab:blue')
                ax_srew.set_title(f'Step-Reward : {self.step_reward:.3f}') #{self.reward_hist[1][-1]:.3f}')

                ax_crew.plot(self.reward_hist[2], color='tab:brown')
                ax_crew.set_title(f'Cummulative-Reward : {self.cummulative_reward:.3f}') #{self.reward_hist[2][-1]:.3f}')
                
                ax_rsum.plot(self.reward_hist[0], color='tab:green')
                ax_rsum.set_title(f'Signal Sum : {self.reward_signal_sum:.3f}') #{self.reward_hist[0][-1]:.3f}')

            if self.enable_imaging:
                ax_xray.set_yticks(self.nr, self.names)
                ax_xray.set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
                ax_xray.grid(axis='both')
                ax_xray.imshow(self.img_xray, aspect=self.img_aspect, cmap= self.render_xray_cmap, vmin= 0, vmax= self.N_BOTS )
                ax_xray.set_title("X-Ray: All Sensors")
                
                ax_dray.set_yticks(self.nr, self.names)
                ax_dray.set_xticks(self.arcTicks[:,0], self.arcTicks[:,1] )
                ax_dray.grid(axis='both')
                ax_dray.imshow(self.img_dray, aspect=self.img_aspect, cmap= self.render_dray_cmap, vmin= 0, vmax= self.SCAN_RADIUS )
                ax_dray.set_title("D-Ray: All Sensors")
        
        #(plt.show() if show_plots else plt.close())
        return fig

    #==============================================================
    def render_state_hook(self, ax):
        pass # <---- use 'ax' to render target points


class ntWorld(World):
        
    def build_observation_space(self):
        # now define observation_space ----------------------------------------------------------------------------
        position_space_info = {  # Position/Direction x,y
                        'dim': 2,
                        'low': (-self.X_RANGE, -self.Y_RANGE), 
                        'high':( self.X_RANGE, self.Y_RANGE), 
                    }
        self.force_clip_low = np.array((-self.X_RANGE, -self.Y_RANGE), dtype= self.STATE_DTYPE)
        self.force_clip_high= np.array((self.X_RANGE, self.Y_RANGE), dtype= self.STATE_DTYPE)
        velocity_space_info = { # velocities dx, dy
                        'dim': 2,
                        'low': (-self.SPEED_LIMIT, -self.SPEED_LIMIT), 
                        'high':( self.SPEED_LIMIT,  self.SPEED_LIMIT), 
                    }
        neighbour_space_info = { # total_neighbours, # occluded neighbours
                        'dim': 2,  
                        'low': (0,             0,         ), 
                        'high':(self.N_BOTS-1, self.N_BOTS-2), 
                    }

        sensor_space_info = { # ( dis, arch, arcl, arcm ) * n_bots-1
                        'dim': (self.N_BOTS-1) * 4,  
                        'low': tuple(np.array([(0, 0, 0, 0) for _ in range(self.N_BOTS-1)]).flatten()), 
                        'high': tuple(np.array([(max(self.X_RANGE, self.Y_RANGE), 2*pi, 2*pi, 2*pi) for _ in range(self.N_BOTS-1)]).flatten()), 
                    }
        # if False:
        #     self.observation_space = get_nspace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
        #                 shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'] + sensor_space_info['dim'], ),
        #                 low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low'] + sensor_space_info['low']  ),
        #                 high=   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] + sensor_space_info['high'] ))
            
        #     self.o_dim = 6 + ((self.N_BOTS-1) * 4)
        # else:
        if self.delta_action_mode:
            shape = (position_space_info['dim']   + velocity_space_info['dim']  + neighbour_space_info['dim'] , )
            low=    (position_space_info['low']     + velocity_space_info['low']   + neighbour_space_info['low']   )
            high =   (position_space_info['high']    + velocity_space_info['high']   + neighbour_space_info['high'] )
            odim = 6
        else: # dont include velocities in observation space
            shape = (position_space_info['dim']   +  neighbour_space_info['dim'] , )
            low=    (position_space_info['low']   +  neighbour_space_info['low']   )
            high =   (position_space_info['high']  + neighbour_space_info['high'] )
            odim = 4


        #self.e_dim = 3
        self.observation_space = get_nspace(n=self.N_BOTS, dtype=self.STATE_DTYPE,
            shape = shape,
            low = low,
            high = high,
            )
        self.o_dim = odim 
        self.base_observation = np.zeros(self.observation_space.shape, self.observation_space.dtype)
        #self.target_observation = self.base_observation[-self.e_dim:]
        self.observation = self.base_observation.reshape((self.N_BOTS, self.o_dim))
        self.initial_observation = np.zeros_like(self.observation)

        # define observation views
        self.xy =           self.observation [:, 0:2] # x,y
        self.x =            self.observation [:, 0:1] # x
        self.y =            self.observation [:, 1:2] # y

        if self.delta_action_mode:
            self.dxy =           self.observation [:, 2:4] # dx,dy
            self.dx =            self.observation [:, 2:3] # dx
            self.dy =            self.observation [:, 3:4] # dy

            self.alln =            self.observation[:, 4:5]
            self.occn =            self.observation[:, 5:6]
        else:
            self.dxy =           np.zeros_like(self.xy) # dx, dy
            self.dx =            self.dxy [:, 0:1] # dx
            self.dy =            self.dxy [:, 1:2] # dy

            self.alln =            self.observation[:, 2:3]
            self.occn =            self.observation[:, 3:4]

        #self.sense =            self.observation[:, 6:].reshape((self.N_BOTS, (self.N_BOTS-1) , 4))
        self.sense =            np.zeros((self.N_BOTS, (self.N_BOTS-1) , 4), dtype=self.STATE_DTYPE)
        
        return


    def reset(self, starting_state=None):

        self.episode += 1
        
        if starting_state is None:
            # reset - choose randomly from known initial state distribution 
            self.initial_state = self.initial_states[self.rng.integers(0, len(self.initial_states))]
        else:
            self.initial_state = starting_state

        if self.initial_state.target_point is None:
            rand_point = np.array( [
                        self.rng.uniform(-self.X_RANGE*.5, self.X_RANGE*.5),
                        self.rng.uniform(-self.Y_RANGE*.5, self.Y_RANGE*.5),
                        ] )

        else:
            rand_point=np.array(self.initial_state.target_point)

        #self.initial_observation*=0 < not nessesroy
        for i,p in  enumerate(self.initial_state.points):
            self.initial_observation[i, 0:2] = (np.array(p[0:2]) - rand_point) + \
                ( self.rng.uniform(-self.initial_state.reset_noise, self.initial_state.reset_noise) \
                 if self.initial_state.reset_noise!=0 else 0.0)

        self.TARGET_RADIUS = self.initial_state.target_radius
        self.TARGET_POINT = np.zeros(shape=(2,), dtype=self.STATE_DTYPE)
        
        self.observation[:] = self.initial_observation # copy state vectors
        #self.target_observation[0:2] = self.TARGET_POINT
        #self.target_observation[2] = self.TARGET_RADIUS 
        self.alive[:]=True # all robots alive flag
        self.n_alive = np.sum(self.alive)
        if self.enable_state_hist: 
            self.state_hist.clear()
            #self.state_hist_count=0
            self.state_hist.append(np.copy(self.xy))
            self.state_hist_count=1
        self.actuator*=0 # reset actuator
        # update state variables----------------------
        self.update_distances()
        self.update_sensor_data()
        #---------------------------------------------

        self.reward_signal[:] = self.get_reward_signal()
        self.reward_signal_sum = np.sum(self.reward_signal)
        self.step_reward = 0.0
        self.cummulative_reward = 0.0
        self.ts=0
        self.terminal = self.is_terminal()
        self.timeout = self.is_timeout()
        self.done = self.terminal or self.timeout
        if self.record_reward_hist:
            self.reward_hist = [ [],[],[]
                #[self.reward_signal_sum ], 
                #[self.step_reward], 
                #[self.cummulative_reward]
                ] # signal sum, current reward, cumm reward
        
        return self.base_observation
    
""" ARCHIVE
class World2(World):

    unit_pull = 0.5 # should be less than speed limit
    def build_vectors(self):
        super().build_vectors()
        #self.unit_vector_target = np.zeros_like(self.xy)

    def step(self, action):
        self.read_actuators(action)
        #--------------------
        # calculate unit vector along the target point
        dt = self.TARGET_POINT - self.xy
        nt = np.linalg.norm(dt, 2, axis=-1, keepdims=True)
        ut = (dt/nt)*self.unit_pull
        self.δxy += ut
        #--------------------
        return self.take_action()




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
        
 
        return fig, "render_sensor_image_local"

"""
