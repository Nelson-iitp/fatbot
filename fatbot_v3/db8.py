import fatbot as fb
from math import pi

isd = dict(

    #  [ 'red', 'blue', 'green', 'gold',   'cyan', 'magenta', 'purple', 'brown' ]
    # wide
    f8T = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20),(-30,0),(30,0)], 
    f8 = [(-20,20),(20,0),(20,20),(-20,-20),(-20,0),(20,-20),(0,-30),(0,30)],
    corners2 = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15),(20,25),(-20,-25)], # 4 dots
    corners1 = [(-15,15),(-20,20),(-25,25),(15,-15),(20,-20),(25,-25),(-10,10),(10,-10)], # diagonal

    # closed
    line1 = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10),(18,10),(-18,10)], # closed outsode circle
    dots4x2 = [(-6,3),(-3,6),(-3,3),(6,-3),(3,-6),(3,-3),(-6,6),(6,-6)],  # packed
    sellipse = [(2,2),(-2,2),(2,-2),(-2,-2),(-5,0),(5,0),(0,4),(0,-4)], # packed

)
isd_keys = list(isd.keys())


def envF(testing, horizon, scheme, delta_reward, permute_states, *states):
    return fb.World(
        seed=                   None, 
        name=                   'world_db8', 
        x_range=                40, 
        y_range=                40, 
        enable_imaging=         False, 
        horizon=                horizon, 
        reward_scheme=          fb.RewardSchemes[scheme], 
        delta_reward=           delta_reward, 
        n_bots=                 8, 
        bot_radius=             1, 
        scan_radius=            80, 
        safe_distance=          3, 
        speed_limit=            1, 
        delta_speed=            0.0, 
        sensor_resolution=      40/pi, 
        target_point=           (0.0, 0.0), 
        target_radius=          10.0, 
        record_reward_hist=     testing, 
    
        # renderarfs
        render_normalized_reward=False,
        render_dpi=48, 
        render_figure_ratio=0.4, 
        render_bounding_width=0.5)\
                .add_initial_states(permute_states, *states)


