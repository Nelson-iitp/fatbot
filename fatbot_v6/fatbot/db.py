from .core import World
from math import pi

class db12:

    isd = dict(

        # wide
        f8T = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20),(-30,0),(30,0),(30,15),(30,-15),(-30,15),(-30,-15)], 
        f8 = [(-20,20),(20,0),(20,20),(-20,-20),(-20,0),(20,-20),(0,-30),(0,30),(-10,25),(10,25),(-10,-25),(10,-25)],
        corners2 = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15),(20,25),(-20,-25),(-20,25),(-20,15),(-25,20),(-15,20)], # 4 dots
        corners1 = [(-15,15),(-20,20),(-25,25),(15,-15),(20,-20),(25,-25),(-10,10),(10,-10),(25,25),(20,20),(-20,-20),(-25,-25)], # diagonal

        # closed
        line1 = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10),(18,10),(-18,10),(-18,-10),(-10,-10),(10,-10),(18,-10)],
        dots4x2 = [(-6,3),(-3,6),(-3,3),(6,-3),(3,-6),(3,-3),(-6,6),(6,-6),(-10,10),(10,-10),(15,-15),(-15,15)],  # packed
        sellipse = [(2,2),(-2,2),(2,-2),(-2,-2),(-5,0),(5,0),(0,4),(0,-4),(5,5),(-5,5),(5,-5),(-5,-5)], # packed
    )
    
    def envF(testing, horizon, scheme, delta_reward, permute_states, *states):
        return World(
            seed=                   None, 
            name=                   'world_db12', 
            x_range=                35, 
            y_range=                35, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          scheme, 
            delta_reward=           delta_reward, 
            n_bots=                 12, 
            bot_radius=             1, 
            scan_radius=            70, 
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            #target_point=           (0.0, 0.0), 
            target_radius=          10.0, 
            record_reward_hist=     testing, 
        
            # render args
            render_normalized_reward=False,
            render_dpi=48, 
            render_figure_ratio=0.4, 
            render_bounding_width=0.5)\
                    .add_initial_states(permute_states, *states)


class db8:

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

    def envF(testing, horizon, scheme, delta_reward, permute_states, *states):
        return World(
            seed=                   None, 
            name=                   'world_db8', 
            x_range=                35, 
            y_range=                35, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          scheme, 
            delta_reward=           delta_reward, 
            n_bots=                 8, 
            bot_radius=             1, 
            scan_radius=            70, 
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            #target_point=           (0.0, 0.0), 
            target_radius=          10.0, 
            record_reward_hist=     testing, 
        
            # renderarfs
            render_normalized_reward=False,
            render_dpi=48, 
            render_figure_ratio=0.4, 
            render_bounding_width=0.5)\
                    .add_initial_states(permute_states, *states)


class db6:
    
    isd = dict(
        # wide
        sixT = [(-20,20),(0,20),(20,20),(-20,-20),(0,-20),(20,-20)], 
        six = [(-20,20),(20,0),(20,20),(-20,-20),(-20,0),(20,-20)],
        circle = [(-17,20),(25,0),(17,20),(-19,-20),(-25,0),(19,-20)], 
        corners2 = [(-15,-20),(25,20),(15,20),(-20,-15),(-25,-20),(20,15)], # 3 dots
        corners1 = [(-15,15),(-20,20),(-25,25),(15,-15),(20,-20),(25,-25)], # diagonal

        # closed
        line1 = [(-10,10),(-14,10),(-6,10),(10,10),(14,10),(6,10)], # closed outsode circle
        dots3x2 = [(-6,3),(-3,6),(-3,3),(6,-3),(3,-6),(3,-3)],  # packed
        sellipse = [(2,2),(-2,2),(2,-2),(-2,-2),(-5,0),(5,0)], # packed
        scircle = [(2,5),(-2,5),(2,-5),(-2,-5),(-5,0),(5,0)], # packed
    )

    def envF(testing, horizon, scheme, delta_reward, permute_states, *states):
        return World(
            seed=                   None, 
            name=                   'world_db6', 
            x_range=                30, 
            y_range=                30, 
            enable_imaging=         False, 
            horizon=                horizon, 
            reward_scheme=          scheme, 
            delta_reward=           delta_reward, 
            n_bots=                 6, 
            bot_radius=             1, 
            scan_radius=            60, 
            safe_distance=          3, 
            speed_limit=            1, 
            delta_speed=            0.0, 
            sensor_resolution=      40/pi, 
            #target_point=           (15.0, 15.0), 
            target_radius=          10.0, 
            record_reward_hist=     testing, 
        
            # renderarfs
            render_normalized_reward=False,
            render_dpi=48, 
            render_figure_ratio=0.4, 
            render_bounding_width=0.5)\
                    .add_initial_states(permute_states, *states)


class db4:

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

    def  envF(testing, horizon, scheme, delta_reward, permute_states, *states):
        return World(
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






