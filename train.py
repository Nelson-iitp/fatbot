# %% section :: imports
""" 
   section :: imports
"""
#import numpy as np
#import matplotlib.pyplot as plt
#import torch.nn as nn
import os
import fatbot as fb
import fatbot.config as fbconfig
# %% section :: parse args 
"""
    section :: parse args 
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cc', type=str, default='common_default', help='common configuration')
parser.add_argument('--tc', type=str, default='rl_default', help='training configuration')
argc = parser.parse_args()

# %% section :: read common configuration
"""
    section :: read common configuration
"""
# ---------------------------------------------------------------------
args_cc=argc.cc
assert len(args_cc)>0, f'requires configuration name!'
assert hasattr(fbconfig, args_cc), f'common configuration [{args_cc}] not found'
cc = fb.FAKE( **getattr(fbconfig, args_cc))
print('\nCommon Configuration:')
for k,v in cc.__dict__.items(): print(f'{k}:\t\t[{v}]')
print('\n')
# ---------------------------------------------------------------------

# swarm environment
db = cc.db 

# initial states
global_isd = cc.global_isd 
initial_state_keys = [db.isd[isd] for isd in global_isd.split(',')]

# horizon
horizon = cc.horizon

# discount
gamma = cc.gamma

# target information
target = cc.target
target_rand = cc.target_rand

# Reward Scheme
reward_scheme = cc.reward_scheme 
delta_reward = cc.delta_reward 
scan_radius = cc.scan_radius
reset_noise = (cc.reset_noise, cc.reset_noise)

# %% section :: build enviroments from common configuration
"""
    section :: build enviroments from common configuration
"""
training_env = db.envF(
  testing=False, 
  target = target,
  target_rand=target_rand,
  scan_radius=scan_radius, 
  reset_noise=reset_noise, 
  horizon=horizon, 
  scheme=reward_scheme, 
  delta_reward=delta_reward, 
  point_lists=initial_state_keys, 
  state_history=False)

validation_env = db.envF(
  testing=False, 
  target = target,
  target_rand=False,
  scan_radius=scan_radius, 
  reset_noise=(0.0, 0.0),  # use zero noise on validation env
  horizon=horizon, 
  scheme=reward_scheme, 
  delta_reward=delta_reward, 
  point_lists=initial_state_keys, 
  state_history=False)

#<---- optinally check
#fb.check_env(training_env) 


# %% section :: read training configuration
"""
    section :: read training configuration
"""
# ---------------------------------------------------------------------
args_tc=argc.tc
assert len(args_tc)>0, f'requires configuration name!'
assert hasattr(fbconfig.RL, args_tc), f'training configuration [{args_tc}] not found'
model_name, model_version, model, total_timesteps = getattr(fb.config.RL, args_tc)(training_env, gamma, test=False)
# ---------------------------------------------------------------------
eval_path = os.path.join(model_name, model_version)
os.makedirs(eval_path, exist_ok=True)
checkpoint_path = os.path.join(eval_path,'checkpoints')
best_model_path = os.path.join(eval_path, 'best_model')
final_model_path = os.path.join(eval_path, 'final_model')
print(f'\nTraining Configuration:\n{model_name=}\n{model_version=}\n{eval_path=}\n')


# %% section :: perform training
"""
    section :: perform training
"""

# note down initial state
#validation_env.reset()
#fig=validation_env.render()
#fig.savefig(os.path.join(eval_path, f'world.png'))

training_start_time = fb.common.now()
print(f'Training @ [{eval_path}]')

eval_callback = fb.EvalCallback(validation_env, 
    best_model_save_path  =  eval_path,
    log_path =               eval_path, 
    eval_freq =              int(total_timesteps/10), 
    n_eval_episodes =        1)

checkpoint_callback = fb.CheckpointCallback(
    save_freq=               int(total_timesteps/10), 
    save_path=               checkpoint_path)
#---------------------------------------------

model.learn(
    total_timesteps=total_timesteps,
    log_interval=int(total_timesteps/10), #int(0.1*total_timesteps)
    callback = fb.CallbackList([checkpoint_callback, eval_callback]) # Create the callback list
    #cb = lambda l, g : print('callback', now(), '\n', l, '\n', g)
)
model.save(final_model_path)
training_end_time = fb.common.now()
print(f'Finished!, Time-Elapsed:[{training_end_time-training_start_time}]')

# %% log results
fr,fs=fb.log_evaluations(os.path.join(eval_path,'evaluations.npz'))
fr.savefig(os.path.join(eval_path, f'episode_rewards.png'))
fs.savefig(os.path.join(eval_path, f'episode_lengths.png'))



