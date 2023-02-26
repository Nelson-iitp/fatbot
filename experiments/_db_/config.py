
from fatbot import PPO
from fatbot.config import PPOCONFIG
from .db import envF, isdF
#import numpy as np
#import matplotlib.pyplot as plt


class C1(PPOCONFIG):
    def __init__(self, alias: str, common_initial_states: str, common_target_radius=12.0) -> None:
        super().__init__(alias, common_initial_states, common_target_radius, envF, isdF)

    def build_rewards(self):
        self.base_reward_scheme = dict(
            dis_target_point=           1.0,
            dis_neighbour=              1.0,
            hull_formed=                1.0,
            dis_target_radius=          1.0,
            all_unsafe=                 1.0,
            all_neighbour=              1.0,
            occluded_neighbour=         1.0,
        )  
        self.aux_reward_scheme = dict(
            dis_target_point=           1.0,
            dis_neighbour=              1.0,
            hull_formed=                1.0,
            dis_target_radius=          1.0,
            all_unsafe=                 1.0,
            all_neighbour=              1.0,
            occluded_neighbour=         1.0,
        )  
        self.final_reward_scheme = dict(
            dis_target_point=           1.0,
            dis_neighbour=              1.0,
            hull_formed=                1.0,
            dis_target_radius=          1.0,
            all_unsafe=                 1.0,
            all_neighbour=              1.0,
            occluded_neighbour=         1.0,
        )  

    #  BASE
    def build_base(self):
        self.base_reset_noise =         2.0
        self.base_delta_reward=         True
        self.base_gamma=                0.95 
        self.base_gae_lambda =          0.95
        self.base_horizon=              500
        self.base_test_horizon=         self.base_horizon

        self.base_model_algo =          PPO 
        self.base_model_name =          f'model_{self.alias}'
        self.base_model_version =       'base'
        self.base_test_scheme =         (1, 0, 0) # (best, final, checkpoints) # select-arg
        self.base_test_episodes =       100

        self.base_total_timesteps = 200_000
        self.base_n_steps=          2048
        self.base_batch_size=       32
        self.base_n_epochs =        10
        self.base_start_lr, self.base_end_lr =  0.0005, 0.0005
        self.base_start_cr, self.base_end_cr =  0.25, 0.25

        self.base_policy_kwargs=dict( #activation_fn=  nn.LeakyReLU, 
                    net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])])
        
    #  AUX
    def build_aux(self):
        self.aux_reset_noise_scalar = 1/8
        self.aux_delta_reward=True
        self.aux_gamma = 0.85
        self.aux_gae_lambda = 0.95
        self.aux_horizon=int(self.base_horizon/3) # select-arg
        self.aux_test_horizon=self.aux_horizon # select-arg


        self.aux_model_algo = PPO 
        self.aux_model_name =    f'model_{self.alias}'
        self.aux_model_version = 'auxi'
        self.aux_test_scheme = (1, 0, 0) # (best, final, checkpoints) # select-arg
        self.aux_test_episodes = 1

        self.aux_total_timesteps = 200_000
        self.aux_n_steps= 2048
        self.aux_batch_size = 64
        self.aux_n_epochs = 10
        self.aux_start_lr, self.aux_end_lr =  0.0005, 0.0005
        self.aux_start_cr, self.aux_end_cr =  0.25, 0.25

        self.aux_policy_kwargs=dict( #activation_fn=  nn.LeakyReLU, 
                    net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])])
        
    # FINAL
    def build_final(self):    
        self.final_episodes = 1 # select-arg
        self.final_record_state_history=10 # how many previous states to store
        self.final_verbose=1
        self.final_delta_reward=True
        self.last_n_steps = (25,25)
        self.last_deltas=(0.005, 0.005)


class C2(C1):
    def build_aux(self):
        super().build_aux()

    def build_base(self):
        super().build_base()
