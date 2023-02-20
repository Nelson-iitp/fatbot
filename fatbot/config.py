#import numpy as np
from .db import *
from .import REMAP
from . import PPO
import torch.nn as nn

common_default={
    'db':               db8,
    'global_isd':       'D',
    'horizon':          500,
    'gamma':            0.99,
    'target':           (0.0, 0.0, 12.0), # x, y, r
    'target_rand':      True,
    'reward_scheme':    'RA',
    'delta_reward':     True,
    'scan_radius':      20.0,
    'reset_noise':      2.0,
}

class RL:

    def rl_default(env, gamma, test=False):
        model_algo =    PPO
        model_name =    'PPO'
        model_version = 'base'

        if test: 
            scheme = 3 # 0=best, 1=final, 2=best_final, 3=best_final_checkpoints
            episodes = 5
            return model_name, model_version, model_algo, (scheme, episodes)
        
        # training timesteps
        total_timesteps = 10_000

        # learning rate scheduling
        start_lr, end_lr =  0.0005, 0.0005
        lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
        def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr

        # clip range scheduling
        start_cr, end_cr =  0.25, 0.25
        cr_mapper=REMAP((-0.2,1), (start_cr, end_cr)) # set clip range schedluer
        def cr_schedule(progress): return cr_mapper.in2map(1-progress) #lr

        # model
        model = model_algo(
                policy=             'MlpPolicy', 
                env=                env, 
                learning_rate =     lr_schedule,
                n_steps=            2048,
                batch_size =        128,
                n_epochs =          10,
                gamma =             gamma,
                gae_lambda=         0.95,   #<==============
                clip_range=         cr_schedule,   #<==============
                clip_range_vf=      None, 
                normalize_advantage=True, 
                ent_coef=           0.0, 
                vf_coef=            0.5, 
                max_grad_norm=      0.5, 
                use_sde=            False, 
                sde_sample_freq=    -1, 
                target_kl=          None, 
                tensorboard_log=    None, 
                create_eval_env=    False, 
                verbose=            1, 
                seed=               None, 
                device=             'cpu', 
                _init_setup_model=  True,
                policy_kwargs=dict(
                                #activation_fn=  nn.LeakyReLU, 
                                net_arch=[dict(
                                    pi=[400, 300], 
                                    vf=[400, 300])])) 
        
        return model_name, model_version, model, total_timesteps


"""
'train_freq':       (1, 'episode'), # (1,'step')
'action_noise_F':   'NormalActionNoise',
'action_noise_A':   dict(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
"""