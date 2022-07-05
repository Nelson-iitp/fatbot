
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
import os, argparse
import numpy as np
#import matplotlib.pyplot as plt
import torch.nn as nn
import fatbot
from fatbot.common import now
from fatbot.core import World
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3 import PPO  as AGENT
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

global_policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                            net_arch=[dict(pi=[256, 256, 256, 256], 
                            vf=[256, 256, 256, 256])] )

def td3train(env, timesteps, save_as, device):
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = AGENT("MlpPolicy", env, action_noise=action_noise,  verbose=1, device=device)
    model.learn(total_timesteps=timesteps, log_interval=int(0.1 * timesteps)/10)
    model.save(save_as)
    print(f'Training is finished. Saved Policy @ [{save_as}]') #return AGENT.load(save_as)
   

def train(env, timesteps, save_as, device):
    model = AGENT("MlpPolicy", env, policy_kwargs=global_policy_kwargs, verbose=1, device= device)
    model.learn(total_timesteps=timesteps, log_interval=int(0.1 * timesteps)/10)
    model.save(save_as)
    print('Training is finished')





if __name__ == '__main__':
    # get the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--xls', type=str,              default='',             help='db/*.xlsx')
    parser.add_argument('--seed', type=int,     default=0,                  help='rng seed')
    parser.add_argument('--delta', type=int,     default=0,                  help='if true, uses delta control mode')
    parser.add_argument('--horizon', type=int,  default=0,                  help='horizon, keep 0 for inf horizon')
    parser.add_argument('--alive', type=int,    default=0,                  help='no of bots alive, keep 0 for all alive')
    parser.add_argument('--timesteps', type=int,        default=0,              help='Total timesteps in training')
    parser.add_argument('--save_as', type=str,          default='',             help='name of policy file')
    parser.add_argument('--device', type=str,          default='cpu',             help='name torch device')
    #parser.add_argument('--render', type=str,      default='',              help='csv render modes, keep blank for no rendering')

    args = parser.parse_args()
    if (not args.xls) or (not args.save_as):
        raise Exception(f'Invalid Args: {args = }')

    # Create Environment from xls 
    env = World(
        xls = os.path.join('db', args.xls), 
        delta_action_mode=bool(args.delta),
        seed=(args.seed if args.seed else None))

    # set meta variables(optinal)
    env.build_meta(
        max_episode_steps=args.horizon,
        min_bots_alive=args.alive,
        target_point = (0.0, 0.0),
        xray_cmap='hot', 
        dray_cmap='copper',
        render_figure_ratio=0.8,
        render_bounding_width=0.1,
        render_sensor_figure_size=10)
    
    check_env(env)
    start_time = now()
    print('Start Training >>[',start_time,']')
    train(env, args.timesteps, args.save_as, args.device)
    end_time = now()
    print('[',end_time,']<< End Training')
    print('__[',end_time - start_time,']__')


"""

#=============================================================================================
# Impoter
#=============================================================================================
import datetime

from zmq import device
now = datetime.datetime.now
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

global_policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                                net_arch=[dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]) #vf = value function

#=============================================================================================
# Training
#=============================================================================================
def main_ddpg(env, total_timesteps, log_interval, save_as, test, device):

    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    
    if not test:
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions), theta=0.15, dt=0.01, initial_noise=None)
        # NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished.')
    else:
        model = DDPG.load(save_as)
        rewart = main_test(env, model, render=test-1)
        print('Testing is Finished. Total-Reward =', rewart)
    
    return

def main_ppo(env, total_timesteps, log_interval, save_as, test, device):

    from stable_baselines3 import PPO

    if not test:
        model = PPO("MlpPolicy", env, policy_kwargs=global_policy_kwargs, verbose=1, device= device)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
    else:    
        model = PPO.load(save_as)
        rewart = main_test(env, model, render=test-1)
        print('Testing is Finished. Total-Reward =', rewart)
    return

def main_sac(env, total_timesteps, log_interval, save_as, test, device):
    from stable_baselines3 import SAC

    if not test:
        model = SAC("MlpPolicy", env, verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
    else:
        model = SAC.load(save_as)
        rewart = main_test(env, model, render=test-1)
        print('Testing is Finished. Total-Reward =', rewart)
   
    return

def main_a2c(env, total_timesteps, log_interval, save_as, test, device):
    from stable_baselines3 import A2C

    if not test:
        model = A2C("MlpPolicy", env,  policy_kwargs=global_policy_kwargs, verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
    else:
        model = A2C.load(save_as)
        rewart = main_test(env, model, render=test-1)
        print('Testing is Finished. Total-Reward =', rewart)
   
    return

def main_td3(env, total_timesteps, log_interval, save_as, test, device):
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    if not test:
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, action_noise=action_noise,  verbose=1, device=device)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
    else:
        model = TD3.load(save_as)
        rewart = main_test(env, model, render=test-1)
        print('Testing is Finished. Total-Reward =', rewart)
   
    return

#=============================================================================================
# Testing
#=============================================================================================
def main_test(env, model, render=0):
    obs = env.reset()
    rewart, done = 0, False
    reward_hist = []
    print('Reset')
    if render>0:
        env.render()
    ts=0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f'{ts}% :: {reward = }, {done = }')
        rewart+=reward
        reward_hist.append((reward, rewart))
        if render>1:
            env.render()
    print('Finished!')
    if render>0:
        env.render()
    
    reward_hist=np.array(reward_hist)
    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    ax[0].plot(reward_hist[:,0], label='Reward', color='tab:purple')
    ax[1].plot(reward_hist[:,1], label='Return', color='tab:green')

    ax[0].legend()
    ax[1].legend()
    plt.show()

    return rewart

"""