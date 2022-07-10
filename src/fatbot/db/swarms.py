#==============================================================
from ..core import Swarm
from math import pi
#==============================================================

# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
""" Swarm- Initial State Distribution (isd) """
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -

GLOBAL_ISD = {
    '4x1' : [  [   (5,5),     (-5,5),     (5,-5),     (-5,-5),     ],  ],
    '4x2' : [  [   (9,8),     (-3,6),     (4,-5),     (-5,-5),     ],  ],
}

# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
""" Swarms """
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -

def swarm_4x(*isd_keys):
    swarm = Swarm(
        name="swarm_4x", 
        x_range=20,
        y_range=20,
        n_bots=4, 
        bot_radius=1, 
        scan_radius=15, 
        safe_distance=1, 
        speed_limit=1,
        delta_speed=0,
        sensor_resolution=40/pi,
        min_bots_alive=0,
        horizon=500,
        )
    if isd_keys:
        for isd in isd_keys:
            swarm.initial_states.extend(GLOBAL_ISD[isd])
    else:
        print(f'[!] WARNING: Swarm has no initial state distribution, this will throw exception later!')
    return swarm

def swarm_4x1():
    return swarm_4x('4x1')

def swarm_4x2():
    return swarm_4x('4x1', '4x2')
#==============================================================