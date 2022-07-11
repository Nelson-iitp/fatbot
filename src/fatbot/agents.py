""" agents.py - abstraction of stable_baselines3 based RL-algorithms - independent of core fatbot components """

import os
from math import inf
import numpy as np
import matplotlib.pyplot as plt

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
        make_video=False,
        video_fps=1,
        starting_state=None,
        ):
    # render_mode = [ 'all', 'env', 'rew', 'sen' ] # keep blank for no rendering
    # save_fig = path or keep blank for no saving
    # use model = ModelTypes[model_type].load(model_path), 
    renderer = env.get_render_handler(
        render_mode=render_mode, save_fig=save_fig, save_dpi=save_dpi, make_video=make_video, video_fps=video_fps)
        
    if model is None:
        print('[!] No model provided - Using random actions')
        model = RandomPolicy(env.action_space)

    # start episodes
    renderer.Start()
    episodes = (episodes if episodes>1 else 1)
    test_history = []
    print(f'\n[++] Begin Epoch: Running for {episodes} episodes')
    for episode in range(episodes):
        cs = env.reset(starting_state=starting_state) # reset
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
    average_return = np.average(test_history[:, 1])
    total_steps = np.sum(test_history[:, 0])
    print(f'[--] End Epoch [{episodes}] episodes :: Avg Return: {average_return}, Total Steps: {total_steps}')
    if episodes>1:
        fig, ax = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(f'Test Results')
        ax[0].plot(test_history[:,0], label='Steps', color='tab:purple')
        ax[1].plot(test_history[:,1], label='Return', color='tab:green')
        ax[0].legend()
        ax[1].legend()
        plt.show()
    return average_return, total_steps

class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.action_space.sample(), None


""" Section: abstract Agent class """



class Agent:

    def __init__(self, model_type, model_name, base_dir, training_env, testing_env) -> None:
        self.training_env = training_env
        self.testing_env = testing_env
        self.model_name = model_name
        self.model_type = model_type
        self.base_dir = base_dir
        self.model_path= os.path.join(self.base_dir, self.model_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self.model=None

    def check(self):
        #check_env(self.testing_env)
        return check_env(self.training_env)
        

    def train(self, learn_args, model_args):
        self.model = TRAIN(env=self.training_env,model_type = self.model_type, model_path= self.model_path, 
                    learn_args=learn_args, model_args=model_args)

    def test(self, episodes=1, steps=0, deterministic=True, render_mode='', 
            save_fig='', save_dpi='figure', make_video=False, video_fps=1, starting_state=None):
        if self.model is None:
            self.load_model()
        return TEST(env=self.testing_env,model=self.model,
            episodes=episodes, steps=steps, deterministic=deterministic, render_mode=render_mode,
            save_fig=(os.path.join(self.base_dir, save_fig) if save_fig else save_fig), 
            save_dpi=save_dpi, make_video=make_video, video_fps=video_fps, starting_state=starting_state)

    def load_model(self):
        self.model = ModelTypes[self.model_type].load(self.model_path)


# ----------------------------------------------------------------------------------------------------
