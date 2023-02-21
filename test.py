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
parser.add_argument('--tc', type=str, default='rl_default', help='testing configuration')
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
testing_env = db.envF(
  testing=True, 
  target = target,
  target_rand=target_rand,
  scan_radius=scan_radius, 
  reset_noise=reset_noise, 
  horizon=horizon, 
  scheme=reward_scheme, 
  delta_reward=delta_reward, 
  point_lists=initial_state_keys, 
  state_history=False)



# %% section :: read training configuration
"""
    section :: read testing configuration
"""
# ---------------------------------------------------------------------
args_tc=argc.tc
assert len(args_tc)>0, f'requires configuration name!'
assert hasattr(fbconfig.RL, args_tc), f'testing configuration [{args_tc}] not found'
model_name, model_version, model_algo, (scheme, episodes) = getattr(fb.config.RL, args_tc)(testing_env, gamma, test=True)
# ---------------------------------------------------------------------
eval_path = os.path.join(model_name, model_version)
assert os.path.exists(eval_path), f'eval path {eval_path} not found!'
checkpoint_path = os.path.join(eval_path,'checkpoints')
best_model_path = os.path.join(eval_path, 'best_model')
final_model_path = os.path.join(eval_path, 'final_model')
test_result_path =  os.path.join(eval_path,'results')
os.makedirs(test_result_path, exist_ok=True)
print(f'\nTesting Configuration:\n{model_name=}\n{model_version=}\n{eval_path=}\n{test_result_path=}')


# %% perform testing

def do_testing(testing_env, model, episodes, model_type, verbose=0):
    average_return, total_steps, sehist, tehist, fi, ff = fb.TEST(
        env=testing_env, 
        model=model, 
        episodes=episodes, 
        steps=65, 
        deterministic=True, 
        render_as='None', 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs=dict(local_sensors=True, reward_signal=True, fill_bots=False, state_hist_marker='o'),
        starting_state=None,
        plot_results=0,
        start_n=0,
        reverb=verbose,
        plot_end_states=True,
        save_states='',
        save_prefix=''
    )
    for i,f in enumerate(fi): f.savefig(os.path.join(test_result_path, f'{model_type}_{i+1}_A.png'))
    for i,f in enumerate(ff): f.savefig(os.path.join(test_result_path, f'{model_type}_{i+1}_Z.png'))
    print(f'\n{average_return=}, {total_steps=}, {sehist=}, {len(tehist)=}\n')


def test_best_model(verbose=0):
    model_type='best_model'
    model_path=os.path.join(eval_path, model_type)
    model = model_algo.load(model_path)
    print(f'Testing @ [{model_path}] for {episodes=}')
    do_testing(testing_env, model, episodes, model_type, verbose)

def test_final_model(verbose=0):
    model_type='final_model'
    model_path=os.path.join(eval_path, model_type)
    model = model_algo.load(model_path)
    print(f'Testing @ [{model_path}] for {episodes=}')
    do_testing(testing_env, model, episodes, model_type, verbose)

def test_checkpoint_model(verbose=0):
    for model_type in os.listdir(checkpoint_path):
        if not model_type.lower().endswith('.zip'): continue

        model_path=os.path.join(checkpoint_path, model_type)
        model = model_algo.load(model_path)
        print(f'Testing @ [{model_path}] for {episodes=}')
        do_testing(testing_env, model, episodes, model_type[:model_type.rfind('.')], verbose)
#%%
# load model
if scheme==0:
    test_best_model()
elif scheme==1:
    test_final_model()
elif scheme==2:
    test_best_model()
    test_final_model()
else:
    test_checkpoint_model()
    test_best_model()
    test_final_model()
