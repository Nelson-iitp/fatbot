#%%
import fatbot as fb
import torch.nn as nn
import numpy as np
import fatbot.db6 as db
import matplotlib.pyplot as plt  
import os

#%%

model_name =            'x6'    #<----- stores data in this folder
model_version =         'base'
model_path =            os.path.join(model_name, model_version)

os.makedirs(model_name, exist_ok=True)
reward_scheme = dict( 
                dis_target_point=   1.0, 
                #dis_neighbour =     1.0,
                #dis_target_radius=  1.0, 
                all_unsafe=         1.0, 
                all_neighbour=      1.0, 
                occluded_neighbour= 2.0, 
                )
delta_reward = True
gamma =                 0.5
horizon =               500
total_timesteps =       500_000 #<--- for training


# learning rate scheduling
start_lr, end_lr = 0.00050, 0.000005
lr_mapper=fb.REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
def lr_schedule(progress):
  progress_precent = 100*(1-progress)
  lr = lr_mapper.in2map(1-progress)
  if int(progress_precent) % 10 == 0:
    print(f'Progress: {progress} ~~> {progress_precent:.3f} %,  {lr = }')  
  return lr #lr_mapper.in2map(1-progress) 


                                    
#%%

# initial state distribution - uniformly sample from all listed states
initial_state_keys =  [np.load(os.path.join(model_name,'base_0_final.npy'))] # [db.isd[db.isd_keys[0]]] #[v for k,v in db.isd.items()] 
permute_states =        False
print(f'Total Initial States: {len(initial_state_keys)}')

# build training env
training_env = db.envF(False, horizon, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)

#<---- optinally check
fb.check_env(training_env) 


#%%
training_env.render()

#%%

# start training
training_start_time = fb.common.now()
print(f'Training @ [{model_path}]')
model = fb.PPO(policy=      'MlpPolicy', 
        env=                training_env, 
        learning_rate =     lr_schedule,
        n_steps=            2048+1024,
        batch_size =        64+32,
        n_epochs =          20,
        gamma =             gamma,
        gae_lambda=         0.95,
        clip_range=         0.25, 
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
                        activation_fn=  nn.LeakyReLU, 
                        net_arch=[dict(
                            pi=[512, 512, 512], 
                            vf=[512, 512, 512])])) #256, 256, 256, 128, 128

model.learn(total_timesteps=total_timesteps,log_interval=int(0.1*total_timesteps))
model.save(model_path)
training_end_time = fb.common.now()
print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')



#%%

# initial state distribution - uniformly sample from all listed states
initial_state_keys =  [np.load(os.path.join(model_name,'base_0_final.npy'))]
permute_states =        False
print(f'Total Initial States: {len(initial_state_keys)}')

# build training env
training_env = db.envF(True, 200, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)


#%%
print(f'Testing @ [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            training_env, 
    model=          fb.PPO.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      'testx6', # use None for no plots, use '' (empty string) to plot inline
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None, # either none or lambda episode: initial_state_index (int)
    plot_results=0,
    start_n=0, # for naming render pngs
    save_state_info=model_path, # call plt.show() if true
    save_both_states=False,
)
print(f'{average_return=}, {total_steps=}')
# %%
