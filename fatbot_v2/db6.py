import fatbot as fb
from math import pi

isd = dict(

    # wide
    sixT = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20)], 
    six = [(-20,20),(20,0),(20,20),(-20,-20),(-20,0),(20,-20)],
    circle = [(-17,19),(25,0),(17,19),(-19,-17),(-25,0),(19,-17)], 
    corners2 = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15)], # 3 dots
    corners1 = [(-15,15),(-20,20),(-25,25),(15,-15),(20,-20),(25,-25)], # diagonal

    # closed
    line1 = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10)], # closed outsode circle
    dots3x2 = [(-6,3),(-3,6),(-3,3),(6,-3),(3,-6),(3,-3)],  # packed
    sellipse = [(2,2),(-2,2),(2,-2),(-2,-2),(-5,0),(5,0)], # packed

)
isd_keys = tuple(isd.keys())

scheme_1=dict( 
        dis_target_point=   1.0, 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        #occlusion_ratio=    1.0,
        )

scheme_2=dict( 
        dis_target_point=   1.0, 
        #dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        #occlusion_ratio=    1.0,
        )

scheme_3=dict( 
        #dis_target_point=   1.0, 
        dis_target_radius=  1.0, 
        all_unsafe=         1.0, 
        all_neighbour=      1.0, 
        occluded_neighbour= 1.0, 
        #occlusion_ratio=    1.0,
        )


def envF(testing, scheme, delta_reward, *state_keys):
    return fb.World(
        name=                   'world', 
        x_range=                30, 
        y_range=                30, 
        enable_imaging=         False, 
        horizon=                2000, 
        seed=                   None, 
        reward_scheme=          scheme, 
        delta_reward=           delta_reward, 
        n_bots=                 6, 
        bot_radius=             1, 
        scan_radius=            20, 
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
        render_bounding_width=0.5).add_initial_states(False, *[isd[k] for k in state_keys])


