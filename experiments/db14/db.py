import os
from fatbot import SwarmState #, pjs, pj

#-------------------------------------
# select based on type of world
from fatbot import World as World       #<----- use world with target in state space
# or
#from fatbot import ntWorld as World     #<----- use world with target NOT in state space
#-------------------------------------

# select world range
x_range, y_range =      50, 50 # will range from -range to +range
n_bots=                 14       #  number of bots in the world
bot_radius=             1.0    # body-radius of bots - (may depend world on x,y range)
scan_radius=            20.0   # scan-radius of bots - (may depend world on x,y range)
safe_distance=          3.0    # safety-radius of bots - (may depend world on x,y range and bot_radius)
speed_limit=            1.0    # maximum allowed speed in each direction (disctance covered in unit time)

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
        delta_speed=            0.0, # if not zero, uses delta-action mode, state-space include velocities as well
        horizon=                0, # max allowed timesteps after which env will stop, may depend of size of world, keep 0 for inf horizon
        min_bots_alive=         0, # min no. bots alive, below which world will terminate # keep zero for all alive condition
        reward_scheme=          reward_scheme,
        delta_reward=           delta_reward, # if true, uses reward difference from two states 


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

def isdF(names, target_radius, reset_noise, validation=False):
    p,f =  os.path.split(__file__)
    isd = os.path.join(p, isd_dir, )
    return [SwarmState(
        target_point= ( (0.0, 0.0) if validation else None), 
        target_radius = target_radius, 
        reset_noise=  ( 0.0 if validation else reset_noise ), 
        points = SwarmState.points_from_img(os.path.join(isd, f'{name}.{isd_ext}'), x_range, y_range) ) for name in names.split(',')]

