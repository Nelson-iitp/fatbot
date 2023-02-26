
from .common import pj, pjs, now, mkdir, REMAP, create_dirs, log_evaluations
from .testing import do_checking, do_testing, do_testing2
from .core import SwarmState
from . import PPO
from .import EvalCallback, CheckpointCallback, CallbackList

from os import listdir
#import numpy as np
#import matplotlib.pyplot as plt


class PPOCONFIG(object):
    def __init__(self, alias:str, common_initial_states:str , common_target_radius:float, 
        envF=None, isdF=None) -> None:
        # COMMON CONFIG
        self.alias = alias #'A'
        self.common_initial_states = common_initial_states #"D,P,S"
        self.common_target_radius = common_target_radius
        self.envF = envF
        self.isdF = isdF
        self.build_rewards()
        self.build_base()
        self.build_aux()
        self.build_final()
   
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
        
    def base_get_states(self, names, validation=False):
        return self.isdF(names, 
            target_radius=self.common_target_radius, 
            reset_noise=self.base_reset_noise,
            validation=validation)
    
    def get_base_env(self, test=False, validation=False):
        return  self.envF(
        alias=f'base_{self.alias}',
        testing=test, # if True, enables recording reward history for plotting
        reward_scheme=self.base_reward_scheme,
        delta_reward=self.base_delta_reward,
        record_state_history=0,
        initial_states=self.base_get_states(names=self.common_initial_states, validation=validation)
        )

    def get_base_model(self, env, test=False):
        if test: return self.base_model_name, self.base_model_version, self.base_model_algo, \
            (self.base_test_scheme, self.base_test_episodes)

        # learning rate scheduling
        start_lr, end_lr = self.base_start_lr, self.base_end_lr      # select-arg
        lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
        def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr

        # clip range scheduling 
        start_cr, end_cr =  self.base_start_cr, self.base_end_cr
        cr_mapper=REMAP((-0.2,1), (start_cr, end_cr)) # set clip range schedluer
        def cr_schedule(progress): return cr_mapper.in2map(1-progress) #lr

        # model
        model = self.base_model_algo(
                policy=             'MlpPolicy',
                env=                env, 
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
        
        return self.base_model_name, self.base_model_version, model, self.base_total_timesteps

    def do_train_base(self):
        base_training_env, base_validation_env = self.get_base_env(), self.get_base_env(validation=True)
        base_training_env.set_horizon(self.base_horizon)
        base_validation_env.set_horizon(self.base_horizon)

        base_model_name, base_model_version, base_model, base_total_timesteps = \
            self.get_base_model (base_training_env, test=False)
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            create_dirs(base_model_name, base_model_version)

        training_start_time = now()
        print(f'Training @ [{base_eval_path}]')

        base_eval_callback = EvalCallback(base_validation_env, 
            best_model_save_path  =  base_eval_path,
            log_path =               base_eval_path, 
            eval_freq =              int(base_total_timesteps/10),  # select-arg
            n_eval_episodes =        1)                             # select-arg

        base_checkpoint_callback = CheckpointCallback(
            save_freq=               int(base_total_timesteps/10),  # select-arg
            save_path=               base_checkpoint_path)

        base_model.learn(
            total_timesteps=base_total_timesteps,
            log_interval=int(base_total_timesteps/10), # select-arg
            callback = CallbackList([base_checkpoint_callback, base_eval_callback]) 

        )
        base_model.save(base_final_model_path)
        training_end_time = now()
        print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')

        fr,fs=log_evaluations(pjs(base_eval_path,'evaluations.npz'))
        fr.savefig(pjs(base_eval_path, f'episode_rewards.png'))
        fs.savefig(pjs(base_eval_path, f'episode_lengths.png'))

    def do_check_base(self, episodes=1, steps=1):
        base_testing_env = self.get_base_env(test=True)
        base_testing_env.set_horizon(self.base_test_horizon)
        chkdir = f'check_{self.alias}'
        mkdir(chkdir)
        do_checking(
            env=base_testing_env,
            episodes=episodes,
            steps=steps,
            verbose=1,
            save_prefix='base', save_path=chkdir,
            save_images=True,
        )

    def do_test_base(self):
        base_testing_env = self.get_base_env(test=True)
        base_testing_env.set_horizon(self.base_test_horizon)

        base_model_name, base_model_version, base_algo, (base_scheme, base_episodes)  = \
            self.get_base_model (None, test=True)
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            create_dirs(base_model_name, base_model_version, make=False)
        
        test_result_path =  pjs(base_eval_path, 'results')
        mkdir(test_result_path)

        def test_sel_model(model_type):
            do_testing(model_type=model_type,
                    testing_env=base_testing_env,
                    eval_path=base_eval_path,
                    model_algo=base_algo,
                    episodes=base_episodes,
                    verbose=0, save_states=test_result_path, save_images=False)
            
        def test_checkpoint_model():
            for model_type in listdir(base_checkpoint_path):
                if model_type.lower().endswith('.zip'): test_sel_model(model_type)

        if base_scheme[0]: test_sel_model('best_model')
        if base_scheme[1]: test_sel_model('final_model')
        if base_scheme[2]: test_checkpoint_model()

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
        
    def aux_get_states(self):
        import torch as tt
        import random

        base_model_name, base_model_version, base_algo, (base_scheme, base_episodes)  = \
            self.get_base_model (None, test=True)
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            create_dirs(base_model_name, base_model_version, make=False)

        base_test_result_path =  pjs(base_eval_path, 'results')

        ipsL = listdir(base_test_result_path)
        initial_states=[]
        for l in ipsL:
            if l.lower().endswith('_final'):
                state = tt.load(pjs(base_test_result_path, l))
                if not state.terminal: 
                    initial_states.append(SwarmState(
                        target_point= state.target_point, 
                        target_radius= state.target_radius, 
                        reset_noise=  state.reset_noise*self.aux_reset_noise_scalar , # dont copy reset noise, # select-arg
                        points = state.points))
        
        assert len(initial_states)>=2, f'aux training requires at least 2 initial states'
        vi = initial_states.pop(random.randint(0, len(initial_states)-1))
        vi.reset_noise=0.0  # validation noise should be zero
        val_initial_states = [vi]
        return initial_states, val_initial_states
        
    def get_aux_env(self, test=False):
        
        kwargs=dict(alias=f'aux_{self.alias}',
        reward_scheme=self.aux_reward_scheme,
        delta_reward=self.aux_delta_reward,
        testing=test, # if True, enables recording reward history for plotting
        record_state_history=0,)
        initial_states, val_initial_states  =  self.aux_get_states()
        if test:
            return  self.envF(initial_states=initial_states,**kwargs)
        else:
            return  self.envF(initial_states=initial_states,**kwargs), \
                    self.envF(initial_states=val_initial_states, **kwargs)

    def get_aux_model(self, env, test=False):
        if test: return self.aux_model_name, self.aux_model_version, self.aux_model_algo, \
            (self.aux_test_scheme, self.aux_test_episodes)

        # learning rate scheduling
        start_lr, end_lr = self.aux_start_lr, self.aux_end_lr      # select-arg
        lr_mapper=REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
        def lr_schedule(progress): return lr_mapper.in2map(1-progress) #lr

        # clip range scheduling 
        start_cr, end_cr =  self.aux_start_cr, self.aux_end_cr
        cr_mapper=REMAP((-0.2,1), (start_cr, end_cr)) # set clip range schedluer
        def cr_schedule(progress): return cr_mapper.in2map(1-progress) #lr

        # model
        model = self.aux_model_algo(
                policy=             'MlpPolicy', 
                env=                env, 
                learning_rate =     lr_schedule,
                n_steps=            self.aux_n_steps, 
                batch_size =        self.aux_batch_size,
                n_epochs =          self.aux_n_epochs,
                gamma =             self.aux_gamma,
                gae_lambda=         self.aux_gae_lambda,
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
                policy_kwargs=self.aux_policy_kwargs) 
        
        return self.aux_model_name, self.aux_model_version, model, self.aux_total_timesteps

    def do_train_aux(self):
        aux_training_env, aux_validation_env = self.get_aux_env()
        aux_training_env.set_horizon(self.aux_horizon)
        aux_validation_env.set_horizon(self.aux_horizon)

        aux_model_name, aux_model_version, aux_model, aux_total_timesteps = \
            self.get_aux_model (aux_training_env, test=False)
        aux_eval_path, aux_checkpoint_path, aux_best_model_path, aux_final_model_path = \
            create_dirs(aux_model_name, aux_model_version)

        training_start_time = now()
        print(f'Training @ [{aux_eval_path}]')

        aux_eval_callback = EvalCallback(aux_validation_env, 
            best_model_save_path  =  aux_eval_path,
            log_path =               aux_eval_path, 
            eval_freq =              int(aux_total_timesteps/10), 
            n_eval_episodes =        1)

        aux_checkpoint_callback = CheckpointCallback(
            save_freq=               int(aux_total_timesteps/10), 
            save_path=               aux_checkpoint_path)

        aux_model.learn(
            total_timesteps=aux_total_timesteps,
            log_interval=int(aux_total_timesteps/10), #int(0.1*total_timesteps)
            callback = CallbackList([aux_checkpoint_callback, aux_eval_callback]) 

        )
        aux_model.save(aux_final_model_path)
        training_end_time = now()
        print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')

        fr,fs=log_evaluations(pjs(aux_eval_path,'evaluations.npz'))
        fr.savefig(pjs(aux_eval_path, f'episode_rewards.png'))
        fs.savefig(pjs(aux_eval_path, f'episode_lengths.png'))


    def do_check_aux(self, episodes=1, steps=1):
        aux_testing_env = self.get_aux_env(test=True)
        aux_testing_env.set_horizon(self.aux_test_horizon)
        chkdir = f'check_{self.alias}'
        mkdir(chkdir)
        do_checking(
            env=aux_testing_env,
            episodes=episodes,
            steps=steps,
            verbose=1,
            save_prefix='auxi', save_path=chkdir,
            save_images=True
        )

    def do_test_aux(self):
        aux_testing_env = self.get_aux_env(test=True)
        aux_testing_env.set_horizon(self.aux_test_horizon)

        aux_model_name, aux_model_version, aux_algo, (aux_scheme, aux_episodes)  = \
            self.get_aux_model (None, test=True)
        aux_eval_path, aux_checkpoint_path, aux_best_model_path, aux_final_model_path = \
            create_dirs(aux_model_name, aux_model_version, make=False)
        
        test_result_path =  pjs(aux_eval_path, 'results')
        mkdir(test_result_path)

        def test_sel_model(model_type):
            do_testing(model_type=model_type,
                    testing_env=aux_testing_env,
                    eval_path=aux_eval_path,
                    model_algo=aux_algo,
                    episodes=aux_episodes,
                    verbose=0, save_states=test_result_path, save_images=False)
            
        def test_checkpoint_model():
            for model_type in listdir(aux_checkpoint_path):
                if model_type.lower().endswith('.zip'): test_sel_model(model_type)

        if aux_scheme[0]: test_sel_model('best_model')
        if aux_scheme[1]: test_sel_model('final_model')
        if aux_scheme[2]: test_checkpoint_model()

    # FINAL
    def build_final(self):    
        self.final_episodes = 1 # select-arg
        self.final_record_state_history=10 # how many previous states to store
        self.final_verbose=1
        self.final_delta_reward=True
        self.last_n_steps = (25,25)
        self.last_deltas=(0.005, 0.005)

    def get_final_env(self):
        return  self.envF(
        alias=f'final_{self.alias}',
        testing=True, # if True, enables recording reward history for plotting
        reward_scheme=self.final_reward_scheme,
        delta_reward=self.final_delta_reward,
        record_state_history=self.final_record_state_history,
        initial_states=self.base_get_states(names=self.common_initial_states, validation=False)
        )

    def do_final_test(self):
        final_testing_env = self.get_final_env() # infinite horizon - auto stops
        #base_testing_env.set_horizon(base_horizon+aux_horizon) #<-------- change here

        base_model_name, base_model_version, base_algo, (base_scheme, base_episodes)  = \
            self.get_base_model (None, test=True)
        base_eval_path, base_checkpoint_path, base_best_model_path, base_final_model_path = \
            create_dirs(base_model_name, base_model_version, make=False)

        aux_model_name, aux_model_version, aux_algo, (aux_scheme, aux_episodes)  = \
            self.get_aux_model (None, test=True)
        aux_eval_path, aux_checkpoint_path, aux_best_model_path, aux_final_model_path = \
            create_dirs(aux_model_name, aux_model_version, make=False)
        
        test_result_path =  pjs(base_model_name, 'results')
        mkdir(test_result_path)
        #full_test_result_path =  pjs(test_result_path, 'full')
        #mkdir(full_test_result_path)
        model1=base_algo.load(base_best_model_path)
        model2=aux_algo.load(aux_best_model_path)
        for e in range(self.final_episodes):
            do_testing2(
                env=final_testing_env,
                model1=model1,
                model2=model2,
                model_type='final',
                make_video=False,
                render_as=pjs(test_result_path, f'{e}'),
                last_n_steps=self.last_n_steps, last_deltas=self.last_deltas,
                verbose=self.final_verbose,
                save_states='', #test_result_path,
                save_images=False
            )
