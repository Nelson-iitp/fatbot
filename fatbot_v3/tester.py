
#%% imports
import fatbot as fb
#import torch.nn as nn
import numpy as np
import db6 as db
import matplotlib.pyplot as plt
import os
#%%
# 1. Choose Reward Scheme 
# <--- doesnt matter since we are not training - only recording
reward_scheme =         'Ro42'
delta_reward =          True
horizon =               200

# 2. Choose RL algorithm
sbalgo =                fb.PPO         #<----- model args depend on this
model_name =            'ppo_model'
model_version =         'base'
model_path =            os.path.join(model_name, model_version)

#%%

  # 3.a. Choose pre-built state
initial_state_keys =    [db.isd['circle']]


#%%

# 4 premute if required
permute_states =        False
print(f'{initial_state_keys}')

#%%

# 5. build testing env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()
np.save(f'{model_path}_test_initial.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_initial.png')
#fb.check_env(testing_env) #<---- optinally check


#%%

print(f'Testing: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      'pposim',
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
)
print(f'{average_return=}, {total_steps=}')
fig=testing_env.render()
fig.savefig(f'{model_path}_test_mid.png')
np.save(f'{model_path}_test_mid.npy',testing_env.observation)

# %%

# 2. Choose RL algorithm
  # 3.a. Choose pre-built state

initial_state_keys =    [
  list(np.load(f'{model_path}_test_mid.npy'))
  ]

# 4 premute if required
permute_states =        False
print(f'{initial_state_keys}')

#%%

# 5. build testing env
  
model_version =         'zen'
model_path =            os.path.join(model_name, model_version)

testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()
np.save(f'{model_path}_test_mid.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_mid.png')


print(f'Testing: [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          sbalgo.load(model_path), #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      'pposim',
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
    start_n=int(total_steps),

)
print(f'{average_return=}, {total_steps=}')
fig.savefig(f'{model_path}_test_final.png')
np.save(f'{model_path}_test_final.npy',testing_env.observation)
# %%

