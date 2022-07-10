""" agents.py - abstraction of stable_baselines3 based RL-algorithms - independent of core fatbot components """

import os
from math import inf
import numpy as np
import matplotlib.pyplot as plt
from .core import RunMode

""" Section: shared training/testing methods using stable_baselines3 """

from stable_baselines3.common.env_checker import check_env # use Tester.check_env(env)
from stable_baselines3 import PPO, DDPG, TD3, SAC, A2C # HER, TRPO


ModelTypes=dict(ppo = PPO, ddpg = DDPG, td3 = TD3, sac = SAC, a2c = A2C)

def TRAIN(env, model_type, model_path, learn_args,model_args ):
    model = ModelTypes[model_type]( env=env, **model_args )
    print(f'[+] Start Training Model [{model_type}] @ [{model_path}]')
    model.learn( **learn_args )
    model.save(model_path)
    print(f'[x] Finished Training Model [{model_type}] @ [{model_path}]')
    return model


def TEST(
        env, 
        model=None, 
        episodes=1, 
        steps=0, 
        deterministic=True, 
        render_mode='', 
        save_fig='', 
        save_dpi='figure', 
        make_video=False
        ):
    # render_mode = [ 'all', 'env', 'rew', 'sen' ] # keep blank for no rendering
    # save_fig = path or keep blank for no saving
    # use model = ModelTypes[model_type].load(model_path), 
    renderer = env.get_render_handler(render_mode=render_mode, save_fig=save_fig, save_dpi=save_dpi, make_video=make_video)
        
    if model is None:
        print('[!] No model provided - Using random actions')
        model = RandomPolicy(env.action_space)

    # start episodes
    renderer.Start()
    episodes = (episodes if episodes>1 else 1)
    test_history = []
    print(f'\n[++] Begin Epoch: Running for {episodes} episodes')
    for episode in range(episodes):
        cs = env.reset() # reset
        done = False
        print(f'\n[+] Begin Episode: {episode+1} of {episodes}')
        

        episode_return = 0.0
        episode_timesteps = 0
        episode_reward_history = []
        episode_max_steps = (steps if steps>0 else inf)

        renderer.Render()  #<--- open renderer and do 1st render
        while (not done) and (episode_timesteps<episode_max_steps):
            action, _ = model.predict(cs, deterministic=deterministic) # action = env.action_space.sample() #print(action)
            cs, rew, done , _ = env.step(action)
            episode_return += rew
            episode_reward_history.append((rew, episode_return))
            episode_timesteps+=1
            print(f'  [{episode_timesteps}/{done}]: Reward: {rew}')
            renderer.Render() 

        print(f'[x] End Episode: {episode+1}] :: Return: {episode_return}, Steps: {episode_timesteps}')
        
        if episode_timesteps>1:
            episode_reward_history=np.array(episode_reward_history)
            fig, ax = plt.subplots(2, 1, figsize=(12,6))
            fig.suptitle(f'Episode: {episode+1}')
            ax[0].plot(episode_reward_history[:,0], label='Reward', color='tab:blue')
            ax[1].plot(episode_reward_history[:,1], label='Return', color='tab:green')
            ax[0].legend()
            ax[1].legend()
            plt.show()
        test_history.append((episode_timesteps, episode_return))
    # end episodes
    renderer.Stop() #<--- close renderer
    test_history=np.array(test_history)
    avg_return = np.average(test_history[:, 1])
    total_steps = np.sum(test_history[:, 0])
    print(f'[--] End Epoch [{episodes}] episodes :: Avg Return: {avg_return}, Total Steps: {total_steps}')
    if episodes>1:
        fig, ax = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(f'Test Results')
        ax[0].plot(test_history[:,0], label='Steps', color='tab:purple')
        ax[1].plot(test_history[:,1], label='Return', color='tab:green')
        ax[0].legend()
        ax[1].legend()
        plt.show()
    return avg_return, total_steps

class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.action_space.sample(), None


""" Section: abstract Agent class """



class Agent:

    def __init__(self, model_type, base_dir, envF) -> None:
        self.envF = envF #<--- a callable
        self.model_type = model_type
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)


    def train(self, model_name, learn_args, model_args):
        print(f'[Agent]:: Training: [{model_name}]')
        return TRAIN(
            env=self.envF(RunMode.training),
            model_type = self.model_type, 
            model_path= os.path.join(self.base_dir, model_name), 
            learn_args=learn_args, model_args=model_args)

    def test(self, model_name, enable_history=True, episodes=1, steps=0, deterministic=True, 
                render_mode='', save_fig='',  save_dpi='figure', make_video=False  ):
        # NOTE: render_mode overrides save_fig arg i.e. is render_mode is blank then save_fig doesnt matter
        run_name = f'{model_name}_{save_fig}'
        fig_save_path = (os.path.join(self.base_dir, run_name) if save_fig else save_fig)
        print(f'[Agent]:: Testing: [{model_name}] @ {run_name}')
        return TEST(
            env=self.envF(RunMode.testing if enable_history else RunMode.no_hist),
            model= (None if ((model_name is None) or (not model_name)) else \
                    ModelTypes[self.model_type].load(os.path.join(self.base_dir, model_name))), 
            episodes=episodes, 
            steps=steps, 
            deterministic=deterministic, 
            render_mode=render_mode, 
            save_fig=fig_save_path, save_dpi=save_dpi, make_video=make_video)




# ----------------------------------------------------------------------------------------------------
