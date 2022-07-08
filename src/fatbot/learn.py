import os
from math import inf
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

class baseTester:
    def __init__(self, env) -> None:
        self.env = env

    def check_env(self):
        return check_env(self.env)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self.env.action_space.sample(), None

    def test(self, model, episodes, steps=0, deterministic=False, render=[], save_fig=''):
        # rendermodes = [ 'env', 'rew', 'xray', 'xray_', 'dray', 'dray_', 'sen_' ]
        
        if render:
            if save_fig:
                os.makedirs(save_fig, exist_ok=True)
                def Render(n):
                    for m in render:
                        for figure, name in self.env.render_modes[m]():
                            if name:  # same as -> if not (figure is None):
                                figure.savefig( os.path.join(save_fig, f'{name}({n}).png') )
                                
            else:
                def Render(n):
                    for m in render:
                        _=self.env.render_modes[m]()
        else:
            def Render(n):
                pass

        if model is None:
            print('No model provided - Using random actions')
            model = self

        # start episodes
        episodes = (episodes if episodes>1 else 1)
        test_history = []
        for episode in range(episodes):
            cs = self.env.reset() # reset
            done = False
            print(f'\n[Begin Episode: {episode+1} of {episodes}]')
            

            episode_return = 0.0
            episode_timesteps = 0
            episode_reward_history = []
            episode_max_steps = (steps if steps>0 else inf)

            Render(episode_timesteps)
            while (not done) and (episode_timesteps<episode_max_steps):
                action, _ = model.predict(cs, deterministic=deterministic) # action = env.action_space.sample() #print(action)
                cs, rew, done , _ = self.env.step(action)
                episode_return += rew
                episode_reward_history.append((rew, episode_return))
                episode_timesteps+=1
                print(f'[{episode_timesteps}/{done}]: Reward: {rew}')
                Render(episode_timesteps)

                    
            print(f'[End Episode: {episode+1}] :: Return: {episode_return}, Steps: {episode_timesteps}')
            if episode_timesteps>1:
                episode_reward_history=np.array(episode_reward_history)
                fig, ax = plt.subplots(2, 1, figsize=(12,6))
                fig.suptitle(f'Episode: {episode+1}')
                ax[0].plot(episode_reward_history[:,0], label='Reward', color='tab:blue')
                ax[1].plot(episode_reward_history[:,1], label='Return', color='tab:green')
                ax[0].legend()
                ax[1].legend()
                plt.show()
        # end episodes
        if episodes>1:
            test_history.append((episode_timesteps, episode_return))
            test_history=np.array(test_history)
            fig, ax = plt.subplots(2, 1, figsize=(12,6))
            fig.suptitle(f'Test Results')
            ax[0].plot(test_history[:,0], label='Steps', color='tab:purple')
            ax[1].plot(test_history[:,1], label='Return', color='tab:green')
            ax[0].legend()
            ax[1].legend()
            plt.show()
        return 

    """
    # -------- General Arg Methods
    def default(self, deter=1, save_fig=''): 
        return self.test(model=None, episodes=1, steps=1, deterministic=deter, render=[], save_fig=save_fig)

    def episode(self, deter=1, save_fig=''):
        return self.test(model=None, episodes=1, steps=0, deterministic=deter, render=['all'], save_fig=save_fig)
    """
class baseTrainer:
    def __init__(self, env) -> None:
        self.env = env
    
    def train_ppo(self, params, args):
        model = PPO( policy=args.policy, env=self.env, **params )
        model.learn(total_timesteps=args.timesteps, log_interval=int(0.01*args.timesteps))
        model.save(args.save_as)
        print(f'PPO: Saved @ {args.save_as}')

