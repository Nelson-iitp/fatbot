# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# Run Experiment Script
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-

import argparse
from math import inf
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from fatbot.common import now, fake
from fatbot.models import WR5o as World
from fatbot.learn import baseTester, baseTrainer
import fatbot.worlds as worlds

from stable_baselines3 import PPO
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
# Experiment Core
# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
BASE_PATH = 'test_WR5o'
POLICY_PATH = os.path.join(BASE_PATH, 'pie')
FIGURE_PATH = os.path.join(BASE_PATH, 'fig')
PPO_DICT = dict(

    ppo_world_dual_test = os.path.join(POLICY_PATH, "ppo_world_dual_test"),

    ppo_world_triple_test = os.path.join(POLICY_PATH, "ppo_world_triple_test"),

    ppo_world_quad_test = os.path.join(POLICY_PATH, "ppo_world_quad_test"),

    ppo_world_x4 = os.path.join(POLICY_PATH, "ppo_world_x4"),

    ppo_world_x5 = os.path.join(POLICY_PATH, "ppo_world_x5"),

    ppo_world_x6 = os.path.join(POLICY_PATH, "ppo_world_x6"),

    ppo_world_x7 = os.path.join(POLICY_PATH, "ppo_world_x7"),

    ppo_world_x8 = os.path.join(POLICY_PATH, "ppo_world_x8"),


    ppo_world_o4 = os.path.join(POLICY_PATH, "ppo_world_o4"),

    ppo_world_o5 = os.path.join(POLICY_PATH, "ppo_world_o5"),

    ppo_world_o6 = os.path.join(POLICY_PATH, "ppo_world_o6"),



)

class Trainer(baseTrainer):

    def __init__(self, env) -> None:
        super().__init__(env)

    def ppo_world_any(self, key):
        print('Training PPO - key :', key, PPO_DICT[key])
        args = dict( timesteps = 50_000,  policy = 'MlpPolicy',  save_as=PPO_DICT[key] )
        params = dict(
            learning_rate=              0.0003, # schedule
            n_steps=                    2048*2, 
            batch_size=                 64*2, 
            n_epochs=                   10, 
            gamma=                      0.99, 
            gae_lambda=                 0.95, 
            clip_range=                 0.2, # schedule
            clip_range_vf=              None, # schedule
            normalize_advantage=        True, 
            ent_coef=                   0.0, 
            vf_coef=                    0.5, 
            max_grad_norm=              0.5, 
            target_kl=                  None,
            verbose=                    0, 
            seed=                       None, 
            device=                     'cpu', 
            policy_kwargs=              dict(
                                            activation_fn=nn.LeakyReLU,
                                            net_arch=[dict(
                                                pi=[256, 256, 256, 256], 
                                                vf=[256, 256, 256, 256])]), 
        )
        self.train_ppo(params, fake(args))



class Tester(baseTester):

    def __init__(self, env) -> None:
        super().__init__(env)

    def ppo_world_any(self, key, steps, deter, savefig):
        #print('Testing PPO - key :', key, PPO_DICT[key])
        self.test(model=PPO.load(PPO_DICT[key]), episodes=1, steps=steps, deterministic=deter, render=['all'], save_fig=savefig)

    def default(self, steps, deter, savefig):
        #print('Testing Random')
        self.test(model=None, episodes=1, steps=steps, deterministic=deter, render=['all'], save_fig=savefig)


# @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-


#-------------------------------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------------------------------
def main(args):
    
    if not hasattr(worlds, args.world):
        raise Exception('World [{}] does not exist!'.format(args.world))

    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(POLICY_PATH, exist_ok=True)
    os.makedirs(FIGURE_PATH, exist_ok=True)

    print('Env-Start @[{}]'.format(now()))
    world = World( **getattr(worlds, args.world) )
    default_meta_params = dict(
        xray_cmap='hot', 
        dray_cmap='copper',
        record_reward_hist=(False if args.trainer else True),
        show_plots=bool(args.plot),
        render_figure_ratio=0.5,
        render_bounding_width=0.1,
        render_sensor_figure_size=10)
    world.build_meta(**default_meta_params)
    print(f'Environment Make: [{world.name}]')
    # should supply either test or train params but not both, test params will preceed
    if args.trainer: # if train args are given, then train (ignore test args)
        print(f' -> Training Mode')
        trainer = Trainer(world)
        trainer.ppo_world_any(args.trainer)
    else: # if train args are not given - test mode
        tester = Tester(world)
        if args.tester:  # if test args are not given, use default
            print(f' -> Policy Testing Mode')
            tester.ppo_world_any(args.tester, args.steps, bool(args.deter), (os.path.join(FIGURE_PATH,args.savefig) if args.savefig else ''))
        else:
            print(f' -> Random Testing Mode')
            tester.default(args.steps, bool(args.deter), (os.path.join(FIGURE_PATH,args.savefig) if args.savefig else ''))
    # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
    print('Env-Stop @[{}]'.format(now()))

if __name__ == '__main__':
    # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-    
    parser = argparse.ArgumentParser()
    parser.add_argument('--world',   type=str, default='default', help='world')
    parser.add_argument('--tester',   type=str, default='', help='test function')
    parser.add_argument('--trainer',   type=str, default='', help='train function')
    parser.add_argument('--savefig',   type=str, default='', help='a path to save figures, keep blank to not save (testing mode only)')
    parser.add_argument('--deter',   type=int, default=1, help='1 for deterministic, 0 for stohcastic (testing mode only)')
    parser.add_argument('--steps',   type=int, default=0, help='0 infinite steps (testing mode only)')
    parser.add_argument('--plot',   type=int, default=1, help='1 to call plt.show() otherwise call plt.close() (testing mode only)')
    main( parser.parse_args() )
    # @-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-@-
    




#-------------------------------------------------------------------------------------------------------
