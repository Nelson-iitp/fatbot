
#%% imports
import fatbot as fb
import torch.nn as nn
import numpy as np
import db4 as db
import matplotlib.pyplot as plt
import os


#%%
# 1. Choose Reward Scheme
reward_scheme =         'Rx42' #<--- change , remove targetting point
delta_reward =          True
global_isds =          [v for k,v in db.isd.items()] #'plus'
#%%

sbalgo =                fb.PPO         #<----- model args depend on this
model_name =            'ppo_model'
os.makedirs(model_name, exist_ok=True)
#%%




"""
# 3.b. Load previous state from disk
load_model_name =     'ppo_model'
load_model_version =  '??'
load_model_path =     os.path.join(load_model_name, load_model_version)
cS =                    'final'
initial_state_keys =    [
  list(np.load(f'{load_model_path}_{cS}.npy'))
  ]
"""












#%% BASE TRAINER

gamma =                 0.99
horizon =               500
model_version =         'base'
model_path =            os.path.join(model_name, model_version)
total_timesteps =       100_000
initial_state_keys =    global_isds # 3.a. Choose pre-built state
permute_states =        False # 4 premute if required
print(f'{initial_state_keys}')

training_env = db.envF(False, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys) # 5. build training env
training_env.reset()
np.save(f'{model_path}_initial.npy',training_env.observation)
fig=training_env.render()
fig.savefig(f'{model_path}_initial.png')
del fig
fb.check_env(training_env) #<---- optinally check

mapper=fb.REMAP((-0.2,1), (0.00050, 0.00005)) # set learn rate schedluer
def lr_schedule(progress):
  progress_precent = 100*(1-progress)
  lr = mapper.in2map(1-progress)
  if int(progress_precent) % 10 == 0:
    print(f'Progress: {progress} ~~> {progress_precent:.3f} %,  {lr = }')  
  return lr

print(f'Training Base: [{model_path}]')
model = sbalgo(policy='MlpPolicy', 
        env=training_env, 
        learning_rate = lr_schedule,
        n_steps= 2048+1024,
        batch_size = 64+32,
        n_epochs = 20,
        gamma = gamma,
        gae_lambda= 0.95,
        clip_range=0.25, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        use_sde=False, 
        sde_sample_freq=- 1, 
        target_kl=None, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='cpu', 
        _init_setup_model=True,
        policy_kwargs=dict(
                        activation_fn=nn.LeakyReLU, 
                        net_arch=[dict(
                            pi=[400, 300, 300, 300], 
                            vf=[400, 300, 300, 300])])) #256, 256, 256, 128, 128

model.learn(total_timesteps=total_timesteps,log_interval=int(0.1*total_timesteps))
model.save(model_path)
print(f'Finished!')

# TESTING - No render, saves last state
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)
print(f'Testing Base: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      None,
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
)
print(f'{average_return=}, {total_steps=}')
np.save(f'{model_path}_final.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_final.png')
del fig





# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @



#%% ZEN TRAINER

gamma =                 0.5
horizon =               100
model_version =         'zen'
model_path =            os.path.join(model_name, model_version)
total_timesteps =       50_000

# 3.b. Load previous state from disk
load_model_name =     'ppo_model'
load_model_version =  'base'
load_model_path =     os.path.join(load_model_name, load_model_version)
cS =                  'final'
initial_state_keys =  [list(np.load(f'{load_model_path}_{cS}.npy'))]
permute_states =        False # 4 premute if required
print(f'{initial_state_keys}')

# 5. build training env
training_env = db.envF(False, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
training_env.reset()
np.save(f'{model_path}_initial.npy',training_env.observation)
fig=training_env.render()
fig.savefig(f'{model_path}_initial.png')
del fig
fb.check_env(training_env) #<---- optinally check

# set learn rate schedluer
mapper=fb.REMAP((-0.2,1), (0.00050, 0.00005))
def lr_schedule(progress):
  progress_precent = 100*(1-progress)
  lr = mapper.in2map(1-progress)
  if int(progress_precent) % 10 == 0:
    print(f'Progress: {progress} ~~> {progress_precent:.3f} %,  {lr = }')  
  return lr


print(f'Training Zen: [{model_path}]')
model = sbalgo(policy='MlpPolicy', 
        env=training_env, 
        learning_rate = lr_schedule,
        n_steps= 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = gamma,
        gae_lambda= 0.95,
        clip_range=0.25, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        use_sde=False, 
        sde_sample_freq=- 1, 
        target_kl=None, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='cpu', 
        _init_setup_model=True,
        policy_kwargs=dict(
                        activation_fn=nn.LeakyReLU, 
                        net_arch=[dict(
                            pi=[400, 300, 300, 300], 
                            vf=[400, 300, 300, 300])])) #256, 256, 256, 128, 128

model.learn(total_timesteps=total_timesteps,log_interval=int(0.1*total_timesteps))
model.save(model_path)
print(f'Finished!')

# TESTING - No render, saves last state
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)
print(f'Testing Zen: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      None,
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
)
print(f'{average_return=}, {total_steps=}')
np.save(f'{model_path}_final.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_final.png')
del fig






# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @
#%%




#%% Testing





horizon =               100 #for each env
model_version =         'base'
model_path =            os.path.join(model_name, model_version)
#initial_state_keys =    [db.isd[global_isd]] # 3.a. Choose pre-built state

load_model_version =         'zen'
load_model_path =            os.path.join(model_name, load_model_version)
initial_state_keys =    [ list(np.load(f'{load_model_path}_test_final.npy')) ]
permute_states =        False # 4 premute if required
print(f'{initial_state_keys}')

# 5. build testing env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()

np.save(f'{model_path}_test_initial.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_initial.png')
#fb.check_env(testing_env) #<---- optinally check
#%%


horizon =               100 #for each env
model_version =         'base'
model_path =            os.path.join(model_name, model_version)
#initial_state_keys =    [db.isd[global_isd]] # 3.a. Choose pre-built state

#load_model_version =    'zen'
#load_model_path =       os.path.join(model_name, load_model_version)
initial_state_keys =    [   [(18,8), (8,16), (4,-3), (-15,3) ]]
#initial_state_keys=global_isds
permute_states =        False # 4 premute if required
print(f'{initial_state_keys}')

# 5. build testing env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()

np.save(f'{model_path}_test_initial.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_initial.png')



#%%

test_ist_n = len(testing_env.initial_states)
print(f'Testing Final - Base: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      None, #'pposim',
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=lambda e: int(e%test_ist_n),
    plot_results=0
)
print(f'{average_return=}, {total_steps=}')
#fig=testing_env.render()
#fig.savefig(f'{model_path}_test_mid.png')
np.save(f'{model_path}_test_mid.npy',testing_env.observation)

#%%
initial_state_keys =    [ list(np.load(f'{model_path}_test_mid.npy')) ]
permute_states =        False
print(f'{initial_state_keys}')
#%%
# switch to zen model  
model_version =         'zen'
model_path =            os.path.join(model_name, model_version)
horizon =               50
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()
np.save(f'{model_path}_test_mid.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_mid.png')


print(f'Testing Final Zen: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      None,#'pposim',
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
    start_n=int(total_steps),

)
print(f'{average_return=}, {total_steps=}')
fig=testing_env.render()
fig.savefig(f'{model_path}_test_final.png')
np.save(f'{model_path}_test_final.npy',testing_env.observation)
# %%

