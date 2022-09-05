import fatbot as fb
from math import pi


isd = dict(
    #-------------------------------------
    cross = [(18,18),(18,-18),(-18,18),(-18,-18)],
    #-------------------------------------
    plus = [(0,18),(0,-18),(18,0),(-18,0)],
    #-------------------------------------
    vline = [(-16,18),(-16,8),(-16,-8),(-16,-18)],
    #-------------------------------------
    hline = [(-16,17),(-6,17),(6,17),(16,17)],
    #-------------------------------------
    dline = [(16,16),(6,6),(-6,-6),(-16,-16)],
    #-------------------------------------
    packed_1 = [(-16,17),(-13,17),(13,-14),(16,-14)],
    #-------------------------------------
    packed_2 = [(-16,-17),(-13,-17),(-13,14),(-16,14)],
    #-------------------------------------
    packed_3 = [(13,-10),(13,-15),(13,-5),(-17,-10)],
    #-------------------------------------
    packed_4 = [(0,10),(0,15),(3,5),(-3,5)],
    #-------------------------------------
    random_1 = [(12,10),(-6,-15),(3,-5),(-16,8)],
    #-------------------------------------
    random_2 = [(10,12),(-15,-6),(-6,6),(-16,8)],
    #-------------------------------------
)

isd_keys = list(isd.keys())

all_states = lambda: [v for k,v in isd.items()] 

def  envF(testing, horizon, scheme, delta_reward, permute_states, *states):
    return fb.World(
        seed=                   None, 
        name=                   'world_db4', 
        x_range=                20, 
        y_range=                20, 
        enable_imaging=         False, 
        horizon=                horizon, 
        reward_scheme=          scheme, 
        delta_reward=           delta_reward, 
        n_bots=                 4, 
        bot_radius=             1, 
        scan_radius=            40, 
        safe_distance=          2.5, 
        speed_limit=            1, 
        delta_speed=            0.0, 
        sensor_resolution=      40/pi, 
        target_radius=          2.0, 
        record_reward_hist=     testing, 
    
        # renderarfs
        render_normalized_reward=False,
        render_dpi=48, 
        render_figure_ratio=0.4, 
        render_bounding_width=0.8)\
                .add_initial_states(permute_states, *states)




