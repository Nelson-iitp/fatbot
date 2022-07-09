from math import pi
#import numpy as np
from .core import Arena, Swarm, World

# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
""" A collection of fatbot worlds """
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -

# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -
""" [4-bots] """
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = -

def env_tester(enable_imaging=True, seed=None):
    arena = Arena(name = "test_arena", x_range=20, y_range=20, horizon=500)
    swarm = Swarm(
        name="test_swarm", 
        n_bots=4, 
        bot_radius=1, 
        scan_radius=15, 
        safe_distance=1, 
        speed_limit=1,
        delta_speed=0,
        sensor_resolution=40/pi,
        min_bots_alive=0)
    swarm.initial_states= [
        [   (5,5),     (-5,5),     (5,-5),     (-5,-5),     ],
    ]

    return World(arena, swarm, enable_imaging=enable_imaging, seed=seed)

