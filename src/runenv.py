
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
import os, argparse
#import numpy as np
#import matplotlib.pyplot as plt
import fatbot
from fatbot.common import now
from fatbot.core import World
from stable_baselines3 import TD3 
from stable_baselines3 import PPO  as AGENT

if __name__ == '__main__':
    print('Env-Run @[{}]'.format(now()))
    
    # get the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--xls', type=str,      default='',                 help='db/*.xlsx')
    parser.add_argument('--seed', type=int,     default=0,                  help='rng seed')
    parser.add_argument('--delta', type=int,     default=0,                  help='if true, uses delta control mode')
    parser.add_argument('--horizon', type=int,  default=0,                  help='horizon, keep 0 for inf horizon')
    parser.add_argument('--alive', type=int,    default=0,                  help='no of bots alive, keep 0 for all alive')
    parser.add_argument('--episodes', type=int,  default=1,                 help='total no of episodes')
    parser.add_argument('--policy', type=str,      default='',              help='name or path for policy load')
    parser.add_argument('--render', type=str,      default='',              help='csv render modes, keep blank for no rendering')
    parser.add_argument('--deterministic', type=int,      default='0',              help='true for deterministic, else stohcastic')

    # parse the args
    args = parser.parse_args()
    if (not args.xls):
        raise Exception(f'Invalid Args: {args.xls = }')
    render_modes = args.render.split(',')
    while '' in render_modes:
        render_modes.remove('')
    print(f'Render Modes: [{len(render_modes)}]: {render_modes}')
    print(f'Policy: {args.policy} ~ deterministic: {args.deterministic}')
    
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
        render_figure_ratio=0.5,
        render_bounding_width=0.1,
        render_sensor_figure_size=10)

    #render_modes=[ 'env', 'rew', 'xray', 'xray_', 'dray', 'dray_', 'sen_' ]
    env.test( 
        model=(AGENT.load(args.policy) if args.policy else None), 
        episodes=args.episodes, 
        deterministic=bool(args.deterministic),
        render_modes=render_modes)

    print('Env-Stop @[{}]'.format(now()))




