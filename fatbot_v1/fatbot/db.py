# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=
from .core import Swarm, World
from math import pi
import itertools
# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=




# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=
""" Swarm - Initial State Distribution """
# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=

# define a global dict containing initial state configurations
GLOBAL_ISD = {
    #-------------------------------------
    # each key has a list of initial states (can be permuted)
    # this is a dictionary of { key : list of [ list of ( 2-tuples ) ] }
    #-------------------------------------
    #'example-key' : [ #<---- key has a collection of states from which env chooses randomly, 
        #[ (x1, y1), (x2, y2), ], #<--- State 1
        #[ (x1, y1), (x2, y2), ], #<--- State 2
        # (each row is one state) containing list of n:(one for each of n-bots) points (2-tuple) -which are initial locations (x,y)
    #],

    #-------------------------------------
    '4x1' : [  
        [   (5,5),     (-5,5),     (5,-5),     (-5,-5),     ],  
        ],
    #-------------------------------------
    '4x2' : [  
        [   (9,8),     (-3,6),     (4,-5),     (-5,-5),     ],  
        ],
    #-------------------------------------
    '4o' : [  
        [   (0,0),     (-2,1),     (2,1),     (0,-2.5),     ],  
        ],
    #-------------------------------------
}

# might want to permute states when all bots are same (not uniquely identifiable)
def permute_states(isd_lists):
    states = []
    for isd_list in isd_lists:
        states.extend(itertools.permutations(isd_list, len(isd_list)))
    return states





# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=
""" Swarms """
# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=

def swarm_4x(permute, *isd_keys):
    swarm = Swarm(
        name=               "swarm_4x", 
        x_range=            20,
        y_range=            20,
        n_bots=             4, 
        bot_radius=         1, 
        scan_radius=        15, 
        safe_distance=      1, 
        speed_limit=        1,
        delta_speed=        0,
        sensor_resolution=  40/pi,
        min_bots_alive=     0,
        horizon=            500,
        target_radius =     5.0,
        )
    if isd_keys:
        if permute:
            for isd in isd_keys:
                swarm.initial_states.extend(  permute_states(GLOBAL_ISD[isd])  )    
        else:
            for isd in isd_keys:
                swarm.initial_states.extend(  GLOBAL_ISD[isd]  )
    else:
        print(f'[!] WARNING: Swarm has no initial state distribution, this will throw exception later!')
    return swarm

#==============================================================





# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=
""" Reward Schemes 
    
    <1> use scheme_*() to return a dictionary of reward signals, like:

        [key in World.reward_data] : [weight (float 0 to 1)]

    <2> All available reward signals are listed below (corresponding callers start with 'RF_')

        def scheme_name(self):
            return dict( 
                    random_reward_pos=  1.0, #<----- this is not useful (only for simulation)
                    random_reward_neg=  1.0, #<----- this is not useful (only for simulation)

                    dis_target_point=   1.0, 
                    dis_target_radius=  1.0, 
                    all_unsafe=         1.0, 
                    all_neighbour=      1.0, 
                    occluded_neighbour= 1.0, 
                    occlusion_ratio=    1.0)

    <3> NOTE: use def render_state_hook(self, ax) to get a hook from renderer

    <4> Instead of inheriting from World class, we can use a reward_scheme_args instead
"""
# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=


def scheme_all():
    return dict( 
        random_reward_pos=  1.0, #<----- this is not useful (only for simulation)
        random_reward_neg=  1.0, #<----- this is not useful (only for simulation)

        dis_target_point=   1.0, 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        occlusion_ratio=    1.0)


# contains all usefull signals (no random signals)
def scheme_default(): 
    return dict(
        dis_target_point=   1.0, 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        occlusion_ratio=    1.0)

# @=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=@=

def scheme_R4x(): 
    return dict( 
        dis_target_point=   1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0)


def scheme_R4o(): 
    return dict( 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0)


def scheme_R5x():      
    return dict( 
        dis_target_point=   1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        occlusion_ratio=    1.0)

def scheme_R5o():      
    return dict( 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        occlusion_ratio=    1.0)
