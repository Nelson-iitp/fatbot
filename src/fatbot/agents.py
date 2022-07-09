import os
from math import inf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from .common import image2video
""" Section: shared training/testing methods using stable_baselines3 """

from stable_baselines3.common.env_checker import check_env # use Tester.check_env(env)
from stable_baselines3 import PPO, DDPG, TD3, SAC, A2C # HER, TRPO
MODEL_TYPES=dict(ppo = PPO, ddpg = DDPG, td3 = TD3, sac = SAC, a2c = A2C)
RANDOM_MODEL_NAME = 'random_run'
def TRAIN(env, model_type, model_path, learn_args, model_args ):
    model = MODEL_TYPES[model_type]( env=env, **model_args )
    model.learn( **learn_args )
    model.save(model_path)
    print(f'Model [{model_type}] Saved: [{model_path}]')
    return model
def TEST(env, model=None, episodes=1, steps=0, deterministic=True, render_mode='', save_fig='', render_dpi='figure'):
    # render_mode = [ 'all', 'env', 'rew', 'sen' ] # keep blank for no rendering
    # save_fig = path or keep blank for no saving
    if not render_mode:
        Render = lambda n: None
    else:
        if save_fig:
            os.makedirs(save_fig, exist_ok=True)
            Render = lambda n: env.render(**(env.render_modes[render_mode]))\
                .savefig( os.path.join(save_fig, f'{n}.png'), transparent=False, dpi=render_dpi   )             
        else:
            Render = lambda n: env.render(**(env.render_modes[render_mode]))
        
    if model is None:
        print('No model provided - Using random actions')
        model = RandomPolicy(env.action_space)

    # start episodes
    episodes = (episodes if episodes>1 else 1)
    test_history = []
    for episode in range(episodes):
        cs = env.reset() # reset
        done = False
        print(f'\n[Begin Episode: {episode+1} of {episodes}]')
        

        episode_return = 0.0
        episode_timesteps = 0
        episode_reward_history = []
        episode_max_steps = (steps if steps>0 else inf)

        Render(episode_timesteps)
        while (not done) and (episode_timesteps<episode_max_steps):
            action, _ = model.predict(cs, deterministic=deterministic) # action = env.action_space.sample() #print(action)
            cs, rew, done , _ = env.step(action)
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
def TESTx(env, model_type, model_path, episodes=1, steps=0, deterministic=True, render_mode='', save_fig='', render_dpi='figure'):
    TEST(env, model = MODEL_TYPES[model_type].load(model_path), 
            episodes=episodes, steps=steps, deterministic=deterministic, 
            render_mode=render_mode, save_fig=save_fig, render_dpi=render_dpi)

class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.action_space.sample(), None



""" Section: abstract Agent class """

class Agent:

    def __init__(self, model_type, base_dir, envF ) -> None:
        self.envF = envF #<--- a callable
        self.model_type = model_type
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        # meta params
        # record_reward_hist=     True,   #<-- set False on training loops  - checked on call to reset(), step() and render()
        # show_plots=             True,   #<-- set False on training loops  - checked on call to render()
        self.default_meta_params = dict(
            xray_cmap=              'hot', 
            dray_cmap=              'copper',
            render_figure_ratio=    0.5,
            render_bounding_width=  0.1,)

    def create(self, **meta_kwargs): 
        env = self.envF()
        env.build_meta(**meta_kwargs)
        print(f'Env Created: [{env.name}]')
        return env

    def train(self, model_name, learn_args, model_args):
        return TRAIN(
            env=self.create( **{**self.default_meta_params, **dict(record_reward_hist=False, show_plots=False)}), 
            model_type = self.model_type, 
            model_path= os.path.join(self.base_dir, model_name), 
            learn_args=learn_args, model_args=model_args)

    def test(self, model_name, show_reward_hist=True, episodes=1, steps=0, deterministic=True, 
                render_mode='', save_fig='', render_dpi='figure', make_video=False  ):
        # NOTE: render_mode overrides save_fig arg i.e. is render_mode is blank then save_fig doesnt matter
        run_name = f'{model_name}_{save_fig}'
        fig_save_path = (os.path.join(self.base_dir, run_name) if save_fig else save_fig)
        TESTx(
            env=self.create( **{
                **self.default_meta_params, 
                **dict(record_reward_hist=show_reward_hist, show_plots=not(bool(fig_save_path)))
                }),
            model_type = self.model_type, 
            model_path= os.path.join(self.base_dir, model_name),
            episodes=episodes, 
            steps=steps, 
            deterministic=deterministic, 
            render_mode=render_mode, 
            save_fig=fig_save_path,
            render_dpi = render_dpi)
        if make_video and fig_save_path and render_mode:
            image2video(fig_save_path)
        return





    def random(self, show_reward_hist=True, episodes=1, steps=0, deterministic=True, 
                render_mode='all', save_fig='', render_dpi='figure', make_video=False ):
        fig_save_path = (os.path.join(self.base_dir, f'{RANDOM_MODEL_NAME}_{save_fig}') if save_fig else save_fig)
        TEST(
            env=self.create( **{
                **self.default_meta_params, 
                **dict(record_reward_hist=show_reward_hist, show_plots=not(bool(fig_save_path)))
                }),
            model= None,
            episodes=episodes, 
            steps=steps, 
            deterministic=deterministic, 
            render_mode=render_mode, 
            save_fig=fig_save_path,
            render_dpi=render_dpi)
        if make_video and fig_save_path and render_mode:
            image2video(fig_save_path, f'{RANDOM_MODEL_NAME}.avi')
        return

class ppoAgent(Agent):

    def __init__(self, base_dir, envF) -> None:
        super().__init__('ppo', base_dir, envF)



    def agent_01(self, test=0, **testargs):
        # testargs = show_reward_hist=True, episodes=1, steps=0, deterministic=True, render_mode='', save_fig='', make_video=False
        if test:
            self.test(model_name = f'agent_007', **testargs)
        else:
            self.train(

                model_name = f'agent_007',

                learn_args = dict(total_timesteps = 10_000, log_interval = int(10_000*0.1)),

                model_args = dict(# env = ? #<----- do not provide env here
                                policy =                    'MlpPolicy',
                                learning_rate=              0.0003, # schedule
                                n_steps=                    1024, #2048, 
                                batch_size=                 64, 
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
                                                                    vf=[256, 256, 256, 256])])))
        
        
