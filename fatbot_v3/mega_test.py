
#%% imports
import fatbot as fb
import torch.nn as nn
import numpy as np
import db8 as db
import matplotlib.pyplot as plt
import os


#%%

sbalgo =                fb.PPO         #<----- model args depend on this
model_name =            'ppo_model'
os.makedirs(model_name, exist_ok=True)
global_isd =            'corners2'



#%% Testing
horizon =               300 #for each env
model_version =         'base'
model_path =            os.path.join(model_name, model_version)
initial_state_keys =    [db.isd[global_isd]] # 3.a. Choose pre-built state
permute_states =        False # 4 premute if required
print(f'{initial_state_keys}')
reward_scheme, delta_reward = 'default', True
# 5. build testing env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()
np.save(f'{model_path}_test_initial.npy',testing_env.observation)
fig=testing_env.render()
fig.savefig(f'{model_path}_test_initial.png')
#fb.check_env(testing_env) #<---- optinally check

print(f'Testing Final - Base: [{model_path}]')
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


initial_state_keys =    [ list(np.load(f'{model_path}_test_mid.npy')) ]
permute_states =        False
print(f'{initial_state_keys}')

# switch to zen model  
model_version =         'zen'
model_path =            os.path.join(model_name, model_version)
horizon =               100
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

