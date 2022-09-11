#%%
#------------------------------
# Imports
#------------------------------
import fatbot as fb
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  
import os

#%%
#------------------------------
# Training Parameters
#------------------------------

db = fb.db.db4         # world model class
print(f'{db=}')
#-----------------------------
model_name =            'x'    #<----- stores data in this folder
model_version =         'base'   #<----- ... saves policy with this name
model_path =            os.path.join(model_name, model_version)
eval_path =             os.path.join(model_name, 'eval')
os.makedirs(model_name, exist_ok=True)
#-----------------------------
total_timesteps =       5_000
horizon =               50
gamma =                 0.5 # discount factor
#-----------------------------
initial_state_list =  list(db.isd.values()) 
permute_states =      False
#-----------------------------
scheme=dict( 
            #dis_target_point=   1.0, 
            dis_neighbour =     1.0,
            #dis_target_radius=  1.0, 
            all_unsafe=         1.0, 
            all_neighbour=      1.0, 
            occluded_neighbour= 2.0, 
            )
delta_reward=True
#-----------------------------

# learning rate scheduling
start_lr, end_lr = 0.00050, 0.000005
lr_mapper=fb.REMAP((-0.2,1), (start_lr, end_lr)) # set learn rate schedluer
def lr_schedule(progress):
  progress_precent = 100*(1-progress)
  lr = lr_mapper.in2map(1-progress)
  if int(progress_precent) % 10 == 0:
    print(f'Progress: {progress} ~~> {progress_precent:.3f} %,  {lr = }')  
  return lr #lr_mapper.in2map(1-progress) 
  #-----------------------------


#%%
#------------------------------
# Training Env Initialize 
#------------------------------
training_env = db.envF(
  False, 
  horizon,
  scheme,
  delta_reward,
  permute_states,
  *initial_state_list )

eval_env = db.envF(
  False, 
  horizon,
  scheme,
  delta_reward,
  permute_states,
  *initial_state_list )

fb.check_env(training_env) #<---- optinally check at least once


#%%
#------------------------------
# Training Model
#------------------------------
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

#%%
#------------------------------
# Training Loop
#------------------------------
training_start_time = fb.common.now()
print(f'Training @ [{model_path}]')
model.learn(
  total_timesteps=total_timesteps,
  callback=None,
  log_interval=int(0.1*total_timesteps), 
  eval_env=eval_env, 
  eval_freq=int(0.01*total_timesteps),
  n_eval_episodes=1, eval_log_path=eval_path)
model.save(model_path)
training_end_time = fb.common.now()
print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')


res = np.load(os.path.join(eval_path, 'evaluations.npz'))
fig, ax = plt.subplots(2, 1, figsize=(16,5))
ax[0].plot(res['timesteps'], res['results'], color='tab:green')
ax[1].plot(res['timesteps'], res['ep_lengths'], color='tab:purple')
#plt.show()
fig.savefig(os.path.join(eval_path, 'evaluations.png'))
res.close()
#%%

#------------------------------
# Testing Env Initialize 
#------------------------------
testing_initial_state_list =  list(db.isd.values()) 
testing_permute_states =      False
testing_horizon = int(horizon/1)
testing_env = db.envF(
  True, 
  testing_horizon,
  scheme,
  delta_reward,
  testing_permute_states,
  *testing_initial_state_list )

render_as = os.path.join(model_name, 'render')

print(f'Testing @ [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          fb.PPO.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      render_as, # use None for no plots, use '' (empty string) to plot inline
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=False, reward_signal=True),
    starting_state=None, # either none or lambda episode: initial_state_index (int)
    plot_results=0,
    start_n=0, # for naming render pngs
    save_state_info=eval_path, # call plt.show() if true
    save_both_states=False,
)
print(f'{average_return=}, {total_steps=}')
# %%
