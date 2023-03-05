import os
from torch import load as load_state
from fatbot import SwarmState #, pjs, pj

#-------------------------------------
# select based on type of world
from fatbot import World as World       #<----- use world with target in state space
# or
#from fatbot import ntWorld as World     #<----- use world with target NOT in state space
#-------------------------------------

# select world range
x_range, y_range =      15, 15 # will range from -range to +range
n_bots=                 4       #  number of bots in the world
bot_radius=             0.4    # body-radius of bots - (may depend world on x,y range)
scan_radius=            15.0   # scan-radius of bots - (may depend world on x,y range)
safe_distance=          1.2    # safety-radius of bots - (may depend world on x,y range and bot_radius)
speed_limit=            0.2    # maximum allowed speed in each direction (disctance covered in unit time)
delta_speed =           0.0     # if not zero, uses delta-action mode, state-space include velocities as well

# select render args
render_dpi=             32
render_figure_ratio=    0.5
render_bounding_width=  0.5

# select data args
isd_dir = 'isd'
isd_ext = 'png'

def envF(
    alias,
    testing,                          
    reward_scheme,         
    delta_reward=True,
    record_state_history=0,
    initial_states=[],
    frozen=False,
    observe_target=True,
    ):
    return World(
        name=                   'world_{}_{}'.format(alias, ('test' if testing else 'train')), 
        seed=                   None,  # for rng
        x_range=                x_range, # x will range from -xrange to +xrange
        y_range=                y_range, # y will range from -yrange to +yrange 

        n_bots=                 n_bots, 
        bot_radius=             bot_radius, 
        scan_radius=            scan_radius, 
        safe_distance=          safe_distance, 
        speed_limit=            speed_limit, 
        delta_speed=            delta_speed, 
        horizon=                0, # max allowed timesteps after which env will stop, may depend of size of world, keep 0 for inf horizon
        min_bots_alive=         0, # min no. bots alive, below which world will terminate # keep zero for all alive condition
        reward_scheme=          reward_scheme,
        delta_reward=           delta_reward, # if true, uses reward difference from two states 
        frozen =                frozen,
        observe_target =        observe_target,

        record_reward_hist=     testing, # should be made true when creating a testing environment 
        enable_imaging=         False, # true to use sensors
        sensor_resolution=      40, # resolution - pixel per unit (of arc length) 2*pi
        record_state_history =  record_state_history, # number of states to record, keep 0 to not record state

        # renderarfs
        render_normalized_reward=   False, # if true, shows reward signal values without assigned weight
        render_dpi=                 render_dpi,  # dpi argument to plt.figure()
        render_figure_ratio=        render_figure_ratio, # determines figsize argument in plt.figure()
        render_bounding_width=      render_bounding_width, # setting the x-y lim on state view 
        ).add_initial_states(*initial_states)

def isdF(names, target_points, target_range, target_radius, reset_noise):
    p,f =  os.path.split(__file__)
    isd = os.path.join(p, isd_dir, )

    has_point = (target_points is not None)
    multi_point = ( hasattr(target_points, '__len__') )
    #target_point = ( ( target_points[i] if  multi_point else target_points ) if has_point else None )
    return [SwarmState(
        target_point = ( ( target_points[i] if  multi_point else target_points ) if has_point else None ), 
        target_range = target_range,
        target_radius = target_radius, 
        reset_noise=  reset_noise , 
        points = SwarmState.points_from_img(os.path.join(isd, f'{name}.{isd_ext}'), x_range, y_range) ) for i,name in enumerate(names.split(','))]

def isdL(path, reset_noise=0.0):

    ipsL = os.listdir(path)
    initial_states=[]
    for l in ipsL:
        if l.lower().endswith('.state'):
            state = load_state(os.path.join(path, l))
            if reset_noise is not None: state.reset_noise=reset_noise
            if not state.terminal: initial_states.append(state)
    return initial_states
