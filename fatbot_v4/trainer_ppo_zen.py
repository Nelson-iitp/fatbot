
#%% imports
import fatbot as fb
import torch.nn as nn
import numpy as np
import db6 as db
import matplotlib.pyplot as plt
import os
#%%
# 1. Choose Reward Scheme
reward_scheme =         'Ro42' #<--- change , remove targetting point
delta_reward =          True
gamma =                 0.5
horizon =               300
# 2. Choose RL algorithm
sbalgo =                fb.PPO         #<----- model args depend on this
model_name =            'ppo_model'
model_version =         'zen'
model_path =            os.path.join(model_name, model_version)
total_timesteps =       50_000

#%%

if False:
  # 3.a. Choose pre-built state
  initial_state_keys =    [
    db.isd['??']
    ]
else:
  # 3.b. Load previous state from disk
  load_model_name =     'ppo_model'
  load_model_version =  'base'
  load_model_path =     os.path.join(load_model_name, load_model_version)
  cS =                    'final'
  initial_state_keys =    [
    list(np.load(f'{load_model_path}_{cS}.npy'))
    ]

#%%

# 4 premute if required
permute_states =        False
print(f'{initial_state_keys}')

#%%

os.makedirs(model_name, exist_ok=True)
# 5. build training env
training_env = db.envF(False, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
training_env.reset()
np.save(f'{model_path}_initial.npy',training_env.observation)
fig=training_env.render()
fig.savefig(f'{model_path}_initial.png')

del fig

fb.check_env(training_env) #<---- optinally check



#%%

# set lear rate schedluer
mapper=fb.REMAP((-0.2,1), (0.00050, 0.00005))
def lr_schedule(progress):
  progress_precent = 100*(1-progress)
  lr = mapper.in2map(1-progress)
  if int(progress_precent) % 10 == 0:
    print(f'Progress: {progress} ~~> {progress_precent:.3f} %,  {lr = }')
  
  return lr


print(f'Training: [{model_path}]')
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
                            pi=[400, 300, 300], 
                            vf=[400, 300, 300])])) #256, 256, 256, 128, 128

model.learn(total_timesteps=total_timesteps,log_interval=int(0.1*total_timesteps))
model.save(model_path)
print(f'Finished!')

#%%
# TESTING - No render, saves last state
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)
print(f'Testing: [{model_path}]')
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

#%%

# NOTE:

"""
model_name/model_version
  initial.npy, final.npy  : last state in training, testing respectively
  initial.png, final.png  : states while training, testing respectively
"""