#==============================================================
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .core import World
#==============================================================

class WR4(World):


    def is_done(self):
        return bool( (self.ts>=self._max_episode_steps) or ( self.n_alive < self.N_BOTS)  )

    def set_reward_signal(self):        
        self.reward_labels =          ['d2Target', 'Unsafe',  'Occluded', 'Naybur']
        range_multiplier = (1 if self.N_BOTS<2 else self.N_BOTS-1)
        reward_contribution =np.array([ 
                # Norm,                                                    weight
                ((((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS),  -1),  # target_dis - lower is better

                (((2*self.N_BOTS*(range_multiplier))),                    -1),  # unsafe_bots- lower is better

                (((self.N_BOTS*(range_multiplier))),                      -1),  # no of occlusions - lower is better

                (((self.N_BOTS*(range_multiplier))),                       1),  # no of neighbours -  higher is better

            ], dtype=self.REWARD_DTYPE)

        self.reward_sign =  reward_contribution[:, 1] # <--- +/- reward
        self.reward_norms = reward_contribution[:, 0] # <--- maximum value of Reward

    def get_reward_signal(self):  # higher is better
         return self.reward_w8 * np.array((
            # sum of distance of all robots rom target - lower is better
            self.reward_signal_distance_from_target(), 

            # no of unsafe distances in dmat - lower is better
            (len(np.where(  (self.dmat<self.SAFE_CENTER_DISTANCE) & (self.dmat>0)  ) [0])/2), 

            # nof of occluded neighbours - lower is better
            np.sum(self.occn),

            # nof of neighbours, higher is better
            np.sum(self.alln), 

        ), dtype=self.REWARD_DTYPE)
        
    def reward_signal_distance_from_target(self): 
        rew = 0.0
        for n in range(self.N_BOTS):
            rew+=np.linalg.norm( self.xy[n, :], 2 ) #self.target_point=0,0 # dead bots will stay at place
        return rew

            

class WR4x(WR4):

    """ Section: Reward Signal """
  
    def set_reward_signal(self):        
        self.reward_labels =          ['d2Target', 'Unsafe',  'Occluded', 'Naybur']
        range_multiplier = (1 if self.N_BOTS<2 else self.N_BOTS-1)
        reward_contribution =np.array([ 
                # Norm,                                                    weight
                ((((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS),  -0.25),  # target_dis - lower is better

                (((self.N_BOTS*(range_multiplier))),                      -1),  # unsafe_bots- lower is better

                (((self.N_BOTS*(range_multiplier))),                      -1),  # no of occlusions - lower is better

                (((self.N_BOTS*(range_multiplier))),                       1),  # no of neighbours -  higher is better

            ], dtype=self.REWARD_DTYPE)

        self.reward_sign =  reward_contribution[:, 1] # <--- +/- reward
        self.reward_norms = reward_contribution[:, 0] # <--- maximum value of Reward



class WR5o(World):
    TARGET_RADIUS = 0.0
    def render_state_handle(self, ax):
        if self.TARGET_RADIUS>0:
            ax.add_patch( # target circle
                Circle(   ( 0, 0 ), ( self.TARGET_RADIUS ),    # 2-tuple center, # float radius
                    color='black', linewidth=0.5, fill=False, linestyle='dashed'))

    def is_done(self):
        return bool( (self.ts>=self._max_episode_steps) or ( self.n_alive < self.N_BOTS)  )

    def set_reward_signal(self):        
        self.reward_labels =          ['d2TargetC', 'dUnsafe', 'dNaybur', 'oPixel', 'nVisible']
        #range_multiplier = (1 if self.N_BOTS<2 else self.N_BOTS-1)
        reward_contribution =np.array([ 
                # Norm,                                                    weight
                ((((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS),  -1),  # target_dis - lower is better

                (((self.N_BOTS*self.SAFE_CENTER_DISTANCE))*2,               -1),  # unsafe_dis- lower is better

                ((((self.X_RANGE**2+self.Y_RANGE**2)**0.5)*self.N_BOTS),  -1),  # sum of interdistances  -  lower is better

                (1,                                                        -2),  # occluded picels -  lower is better

                (self.N_BOTS*self.N_BOTS,                      2),  # all - occ = visible = higer is better

            ], dtype=self.REWARD_DTYPE)

        self.reward_sign =  reward_contribution[:, 1] # <--- +/- reward
        self.reward_norms = reward_contribution[:, 0] # <--- maximum value of Reward

    def get_reward_signal(self):  # higher is better
         return self.reward_w8 * np.array((
            # sum of distance of all robots rom target - lower is better
            self.reward_signal_distance_from_target(), 

            # no of unsafe distances in dmat - lower is better
            np.sum(self.dmat[np.where(  (self.dmat<self.SAFE_CENTER_DISTANCE) & (self.dmat>0) ) ]), 

            # distance of neighbours -  lower is better distance_key = 9
           # np.sum(self.sensor_data[:, :, 9]),
           np.sum(self.dmat)/2,


            # occluded picels
            self.reward_signal_occlusion(),
            
            np.sum(self.alln-self.occn), 

        ), dtype=self.REWARD_DTYPE)
        
    
    def reward_signal_distance_from_target(self): 
        rew = 0.0
        for n in range(self.N_BOTS):
            rew+= np.abs(self.TARGET_RADIUS-(np.linalg.norm( self.xy[n, :], 2 ) )) #self.target_point=0,0 # dead bots will stay at place
        return rew

    def reward_signal_occlusion(self):
        rew=0.0
        for n in range(self.N_BOTS):
            xray = self.img_xray[n]
            rew+=(len(np.where(xray>1)[0])/self.SENSOR_IMAGE_SIZE)
        return rew






























#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
""" NOTE:

"""
#--------------------------------------------------------------------------------------------------------------