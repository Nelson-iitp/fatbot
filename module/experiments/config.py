
#from . import SubprocVecEnv
from fatbot.common import pj, pjs, now, mkdir, REMAP, log_evaluations
from fatbot.core import SwarmState
from fatbot import PPO
from fatbot import EvalCallback, CheckpointCallback, CallbackList

from .testing import do_checking, do_testing, do_testing2, do_testing3

import os
#import numpy as np
#import matplotlib.pyplot as plt

def create_dirs(model_dir, model_version, make=True):
    eval_path = pjs(model_dir, model_version)
    if make:
        mkdir(eval_path)
    else:
        assert os.path.exists(eval_path), f'eval path {eval_path} not found!'
    checkpoint_path = pjs(eval_path,'checkpoints')
    best_model_path = pjs(eval_path, 'best_model')
    final_model_path = pjs(eval_path, 'final_model')
    print(f'\nConfiguration:\n{model_dir=}\n{model_version=}\n{eval_path=}\n')
    return eval_path, checkpoint_path, best_model_path, final_model_path



class PPOCONFIG(object):


    def create_dirs(self, make):
        return create_dirs(self.base_dir, self.base_model_version, make=make)

    def __repr__(self) -> str: return self.alias
    def __str__(self) -> str: return self.alias
    
    def __init__(self, alias:str, common_initial_states:str , 
        results_dir = None, envF=None, isdF=None, isdL=None) -> None:
        # COMMON CONFIG
        
        self.alias = alias #'A'
        self.common_initial_states = common_initial_states #"D,P,S"
        self.envF = envF
        self.isdF = isdF
        self.isdL = isdL

        self.results_dir = results_dir
        self.base_dir = pjs(self.results_dir, self.alias)
        self.check_dir = pjs( self.base_dir, f'check')
        mkdir(self.base_dir)

        self.build_rewards()
        self.build_base()
   
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
        self.base_delta_reward=         True

    def build_base(self):
        self.check_state_history = 20
        self.base_reset_noise =         2.0
        self.val_reset_noise =          0.0
        self.base_target_radius =       3.0
        self.base_target_range = ((0.5, 0.5), (0.5, 0.5))
        self.base_observe_target = True
        self.base_target_points = None

        self.base_gamma=                0.95 
        self.base_gae_lambda =          0.95
        self.base_horizon=              500
        self.base_test_horizon=         self.base_horizon

        self.base_model_algo =          PPO 
        self.base_model_version =       'base'
        self.base_test_scheme =         (1, 0, 0) # (best, final, checkpoints) # select-arg
        self.base_test_episodes =       10

        self.base_total_timesteps = 200_000
        self.base_n_steps=          2048
        self.base_batch_size=       32
        self.base_n_epochs =        10
        self.base_start_lr, self.base_end_lr =  0.0005, 0.0005
        self.base_start_cr, self.base_end_cr =  0.25, 0.25

        self.base_n_eval_episodes = 1
        self.base_n_eval_times = 10

        
        self.base_policy_kwargs=dict( #activation_fn=  nn.LeakyReLU, 
                    net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])])
        
    
    def get_base_env(self, test=False, record_state_history=0, frozen=False, validation=False):
        base_env = self.envF(
            alias=f'base_{self.alias}',
            testing=test, # if True, enables recording reward history for plotting
            reward_scheme=self.base_reward_scheme,
            delta_reward=self.base_delta_reward,
            record_state_history=record_state_history,
            initial_states=self.isdF(
                names=self.common_initial_states,
                target_points=self.base_target_points,
                target_range= self.base_target_range,
                target_radius = self.base_target_radius,
                reset_noise = self.val_reset_noise if validation else self.base_reset_noise),
            frozen=frozen,
            observe_target=self.base_observe_target

        )
        base_env.set_horizon(self.base_test_horizon if test else self.base_horizon)
        return base_env

    def get_base_model(self):

        # learning rate scheduling
        start_lr, end_lr = self.base_start_lr, self.base_end_lr      # select-arg
        lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
        def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr

        # clip range scheduling 
        start_cr, end_cr =  self.base_start_cr, self.base_end_cr
        cr_mapper=REMAP((-0.2,1), (start_cr, end_cr)) # set clip range schedluer
        def cr_schedule(progress): return cr_mapper.in2map(1-progress) #lr

        # model
        # Create the vectorized environment
        #venv = SubprocVecEnv([self.get_base_env() for _ in range(self.base_num_cpu)])
        return self.base_model_algo(
                policy=             'MlpPolicy',
                env=                self.get_base_env(), 
                learning_rate =     lr_schedule,
                n_steps=            self.base_n_steps, 
                batch_size =        self.base_batch_size, 
                n_epochs =          self.base_n_epochs, 
                gamma =             self.base_gamma,
                gae_lambda=         self.base_gae_lambda, 
                clip_range=         cr_schedule,   
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
                policy_kwargs=self.base_policy_kwargs) 
        

    def do_train_base(self, checkpoint=False):
        base_validation_env = self.get_base_env(validation=True)

        base_model = self.get_base_model()
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            self.create_dirs(make=True)

        training_start_time = now()
        print(f'Training @ [{base_eval_path}]')

        base_eval_callback = EvalCallback(base_validation_env, 
            best_model_save_path  =  base_eval_path,
            log_path =               base_eval_path, 
            eval_freq =              int(self.base_total_timesteps/self.base_n_eval_times),  # select-arg
            n_eval_episodes =        self.base_n_eval_episodes)                             # select-arg

        base_checkpoint_callback = CheckpointCallback(
            save_freq=               int(self.base_total_timesteps/10),  # select-arg
            save_path=               base_checkpoint_path)
 
        base_callbacks = [base_eval_callback]
        if checkpoint: base_callbacks.append(base_checkpoint_callback)
        base_model.learn(
            total_timesteps=self.base_total_timesteps,
            log_interval=int(self.base_total_timesteps/self.base_n_eval_times), # select-arg
            callback = CallbackList(base_callbacks)

        )
        base_model.save(base_final_model_path)
        training_end_time = now()
        print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')

        fr,fs=log_evaluations(pjs(base_eval_path,'evaluations.npz'))
        fr.savefig(pjs(base_eval_path, f'episode_rewards.png'))
        fs.savefig(pjs(base_eval_path, f'episode_lengths.png'))

    def do_check_base(self, episodes=1, steps=1, use_random_actions=False, verbose=1):
        mkdir(self.check_dir)
        do_checking(
            use_random_actions=use_random_actions,
            env= self.get_base_env(test=True, record_state_history=self.check_state_history),
            episodes=episodes,
            steps=steps,
            verbose=verbose,
            save_prefix='check', save_path=self.check_dir,
            save_images=True,
        )

    def do_test_base(self):
        base_testing_env = self.get_base_env(test=True)
        

        #base_algo, (base_scheme, base_episodes)  = self.get_base_model (test=True)
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            self.create_dirs(make=False)
        
        test_result_path =  pjs(base_eval_path, 'results')
        mkdir(test_result_path)

        def test_sel_model(model_type):
            do_testing(model_type=model_type,
                    testing_env=base_testing_env,
                    eval_path=base_eval_path,
                    model_algo=self.base_model_algo,
                    episodes=self.base_test_episodes,
                    verbose=0, save_states=test_result_path, save_images=True)
            
        def test_checkpoint_model():
            for model_type in os.listdir(base_checkpoint_path):
                if model_type.lower().endswith('.zip'): test_sel_model(model_type)

        if self.base_test_scheme[0]: test_sel_model('best_model')
        if self.base_test_scheme[1]: test_sel_model('final_model')
        if self.base_test_scheme[2]: test_checkpoint_model()

