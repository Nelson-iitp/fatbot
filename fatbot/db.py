import fatbot as fb
from math import pi

global_render_dpi = 48
global_render_figure_ratio = 0.5
global_render_bounding_width = 0.5
class db6:
    isd = dict(

        #  [ 'red', 'blue', 'green', 'gold',   'cyan', 'magenta', 'purple', 'brown' ]
        # wide
        S = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20)], 
        D = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15)], # 3 dots
        P = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10)], # closed outsode circle

    )
    isd_keys = list(isd.keys())


    def envF(testing, scan_radius,  reset_noise, horizon, scheme, delta_reward, point_list, state_history):
        return fb.World(
            seed=                   None, 
            name=                   'world_db6', 
            x_range=                30, 
            y_range=                30, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          fb.RewardSchemes[scheme], 
            delta_reward=           delta_reward, 
            n_bots=                 6, 
            bot_radius=             1, 
            scan_radius=            scan_radius, 
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            target_point=           (0.0, 0.0), 
            target_radius=          10.0, 
            record_reward_hist=     testing, 
            reset_noise =           reset_noise,
            state_history = state_history,
        
            # renderarfs
            render_normalized_reward=False,
            render_dpi=global_render_dpi, 
            render_figure_ratio=global_render_figure_ratio, 
            render_bounding_width=global_render_bounding_width)\
                    .add_initial_states(point_list)


class db8:
        
    isd = dict(

        #  [ 'red', 'blue', 'green', 'gold',   'cyan', 'magenta', 'purple', 'brown' ]
        # wide
        S = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20),(-30,0),(30,0)], 
        #f8 = [(-20,20),(20,0),(20,20),(-20,-20),(-20,0),(20,-20),(0,-30),(0,30)],
        D = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15),(20,25),(-20,-25)], # 4 dots
        #corners1 = [(-15,15),(-20,20),(-25,25),(15,-15),(20,-20),(25,-25),(-10,10),(10,-10)], # diagonal

        # closed
        P = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10),(18,10),(-18,10)], # closed outsode circle
        #dots4x2 = [(-6,3),(-3,6),(-3,3),(6,-3),(3,-6),(3,-3),(-6,6),(6,-6)],  # packed
        #sellipse = [(2,2),(-2,2),(2,-2),(-2,-2),(-5,0),(5,0),(0,4),(0,-4)], # packed

    )
    isd_keys = list(isd.keys())


    def envF(testing, scan_radius,  reset_noise, horizon, scheme, delta_reward, point_list, state_history):
        return fb.World(
            seed=                   None, 
            name=                   'world_db8', 
            x_range=                35, 
            y_range=                35, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          fb.RewardSchemes[scheme], 
            delta_reward=           delta_reward, 
            n_bots=                 8, 
            bot_radius=             1, 
            scan_radius=            scan_radius, #20
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            target_point=           (0.0, 0.0), 
            target_radius=          12.0, 
            record_reward_hist=     testing, 
            reset_noise =           reset_noise,
            state_history = state_history,
            # renderarfs
            render_normalized_reward=False,
            render_dpi=global_render_dpi, 
            render_figure_ratio=global_render_figure_ratio, 
            render_bounding_width=global_render_bounding_width)\
                    .add_initial_states(point_list)


class db10:
        
    isd = dict(

        #  [ 'red', 'blue', 'green', 'gold',   'cyan', 'magenta', 'purple', 'brown' ]
        # wide
        D = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15),(20,25),(-20,-25),(28,12),(-28,-12)], # 4 dots
        S = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20),(-30,0),(30,0), (10,0),(-10,0)], 
        P = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10),(18,10),(-18,10),(24, 15), (-24, 15)], # closed outsode circle


    )
    isd_keys = list(isd.keys())


    def envF(testing, scan_radius,  reset_noise, horizon, scheme, delta_reward, point_list, state_history):
        return fb.World(
            seed=                   None, 
            name=                   'world_db10', 
            x_range=                35, 
            y_range=                35, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          fb.RewardSchemes[scheme], 
            delta_reward=           delta_reward, 
            n_bots=                 10, 
            bot_radius=             1, 
            scan_radius=            scan_radius, 
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            target_point=           (0.0, 0.0), 
            target_radius=          15.0, 
            record_reward_hist=     testing, 
            reset_noise =           reset_noise,
            state_history = state_history,
            # renderarfs
            render_normalized_reward=False,
            render_dpi=global_render_dpi, 
            render_figure_ratio=global_render_figure_ratio, 
            render_bounding_width=global_render_bounding_width)\
                    .add_initial_states(point_list)


