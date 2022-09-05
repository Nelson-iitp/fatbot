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
                dis_neighbour =     1.0,
                dis_target_radius=  1.0, 
                all_unsafe=         2.0, 
                all_neighbour=      1.0, 
                occluded_neighbour= 2.0, 
                )
delta_reward = True
gamma =                 0.95
horizon =               500



# initial state distribution - uniformly sample from all listed states
initial_state_keys =    db.all_states() # [db.isd[db.isd_keys[0]]]  #[v for k,v in db.isd.items()] 
permute_states =        True
nos_initial_states=len(initial_state_keys)
print(f'Total Initial States: {nos_initial_states}')

# build testing_env
testing_env = db.envF(True, horizon, reward_scheme, delta_reward, 
                        permute_states, *initial_state_keys)


print(f'Testing @ [{model_path}]')
average_return, total_steps = fb.TEST(
    env=            testing_env, 
    model=          fb.PPO.load(model_path), #<---- use None for random
    episodes=       11,
    steps=          0,
    deterministic=  True,
    render_as=      None, # use None for no plots, use '' (empty string) to plot inline
    save_dpi=       'figure',
    make_video=     False,
    video_fps=      2,
    render_kwargs=dict(local_sensors=True, reward_signal=True),
    starting_state=lambda ep: ep, # either none or lambda episode: initial_state_index (int)
    plot_results=0,
    start_n=0, # for naming render pngs
    save_state_info=model_path, # call plt.show() if true
    save_both_states=False,
)
print(f'{average_return=}, {total_steps=}')


#%%