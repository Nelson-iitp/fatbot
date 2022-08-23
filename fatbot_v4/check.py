
#%% imports
import fatbot as fb
#import torch.nn as nn
import numpy as np
import db4 as db
import matplotlib.pyplot as plt
import os
#%%
# 1. Choose Reward Scheme 
# <--- doesnt matter since we are not training - only recording
reward_scheme =         'Rx42'
delta_reward =          True

#%%


if True:
  # 3.a. Choose pre-built state
  initial_state_keys =    [
    db.isd['cross']
    ]
else:
  # 3.b. Load previous state from disk
  load_model_name =     'ppo_model'
  load_model_version =  'v1'
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
horizon=10
# 5. build testing env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, permute_states, *initial_state_keys)
testing_env.reset()
_=testing_env.render()
plt.show()
#fb.check_env(testing_env) #<---- optinally check


#%%

average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          None, #<---- use None for random
    episodes=       1,
    steps=          0,
    deterministic=  True,
    render_as=      '',
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=None,
    plot_results=0,
    target_points=[(12,0),]
)
print(f'{average_return=}, {total_steps=}')
#fig=testing_env.render()
#plt.show()

# %%
