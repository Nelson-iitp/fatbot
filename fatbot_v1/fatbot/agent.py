""" agent.py - abstraction of stable_baselines3 based RL-algorithms - independent of core fatbot components """

import os
from math import inf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

""" Section: shared training/testing methods using stable_baselines3 """

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env # use Tester.check_env(env)
from stable_baselines3 import PPO, DDPG, TD3, SAC, A2C, DQN # HER, TRPO


ModelTypes=dict(ppo = PPO, ddpg = DDPG, td3 = TD3, sac = SAC, a2c = A2C, dqn = DQN)

class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.action_space.sample(), None


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
        save_as='', 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        starting_state=None,
        plot_results=0,
        ):
    # render_mode = [ 'all', 'env', 'rew', 'sen' ] # keep blank for no rendering
    # save_as = path or keep blank for no saving
    # use model = ModelTypes[model_type].load(model_path), 
    renderer = env.get_render_handler(
        render_mode=render_mode, save_as=save_as, save_dpi=save_dpi, make_video=make_video, video_fps=video_fps)
    episode_max_steps = (steps if steps>0 else inf)
    print(f'[.] Testing for [{episodes}] episodes @ [{episode_max_steps}] steps')
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
        
        if (plot_results>1) and (episode_timesteps>1):
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
    if (plot_results>0) and (episodes>1):
        fig, ax = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(f'Test Results')
        ax[0].plot(test_history[:,0], label='Steps', color='tab:purple')
        ax[1].plot(test_history[:,1], label='Return', color='tab:green')
        ax[0].legend()
        ax[1].legend()
        plt.show()
    return average_return, total_steps



""" Section: abstract Agent class """

class Agent:

    def __init__(self, model_type, model_name, model_args, base_dir, training_env, testing_env) -> None:
        self.training_env = training_env
        self.testing_env = testing_env
        self.model_name = model_name
        self.model_type = model_type
        self.base_dir = base_dir
        self.model_path= os.path.join(self.base_dir, self.model_name)
        self.model_args=  model_args 
        self.model=None
        
    def check(self):
        #check_env(self.testing_env)
        return check_env(self.training_env)
        
    def train(self, total_timesteps):
        os.makedirs(self.base_dir, exist_ok=True)
        self.model = TRAIN(env=self.training_env, model_type = self.model_type, model_path= self.model_path, 
                    learn_args={**DEFAULT_LEARN_ARGS, **dict(total_timesteps=total_timesteps, log_interval=total_timesteps)},
                    model_args=(DEFAULT_MODEL_ARGS[self.model_type] if (self.model_args is None) else self.model_args) )

    def test(self, episodes=1, steps=0, deterministic=True, render_mode='', 
            save_as='', save_dpi='figure', make_video=False, video_fps=1, starting_state=None, plot_results=0):
        if self.model is None:
            self.load_model()
        return TEST(env=self.testing_env,model=self.model,
            episodes=episodes, steps=steps, deterministic=deterministic, render_mode=render_mode,
            save_as=(os.path.join(self.base_dir, save_as) if save_as else save_as), 
            save_dpi=save_dpi, make_video=make_video, video_fps=video_fps, starting_state=starting_state, plot_results=plot_results)

    def load_model(self):
        self.model = ModelTypes[self.model_type].load(self.model_path)

""" Section: sb3 defaults """


DEFAULT_LEARN_ARGS = dict(
        total_timesteps=10,
        log_interval=10,
        callback=None, 
        eval_env=None, 
        eval_freq=- 1, 
        n_eval_episodes=5, 
        eval_log_path=None, 
        reset_num_timesteps=True)

DEFAULT_POLICY_KWARG = dict(
        activation_fn=nn.LeakyReLU, 
        net_arch=[dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])])

DEFAULT_MODEL_ARGS = dict(

    ppo = dict(
        policy = 'MlpPolicy',
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        use_sde=False, 
        sde_sample_freq=- 1, 
        target_kl=None, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


    sac = dict(
        policy='MlpPolicy',
        learning_rate=0.0003, 
        buffer_size=1000000, 
        learning_starts=100, 
        batch_size=256, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=1,
        gradient_steps=1, 
        action_noise=None, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        ent_coef='auto', 
        target_update_interval=1, 
        target_entropy='auto', 
        use_sde=False, 
        sde_sample_freq=- 1, 
        use_sde_at_warmup=False, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


    a2c = dict(
        policy='MlpPolicy',
        learning_rate=0.0007, 
        n_steps=5, 
        gamma=0.99, 
        gae_lambda=1.0, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
        rms_prop_eps=1e-05, 
        use_rms_prop=True, 
        use_sde=False, 
        sde_sample_freq=- 1, 
        normalize_advantage=False, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None,
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


    ddpg = dict(
        policy='MlpPolicy',
        learning_rate=0.001, 
        buffer_size=1000000, 
        learning_starts=100, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=(1, 'episode'), 
        gradient_steps=- 1, 
        action_noise=None, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


    td3 = dict(
        policy='MlpPolicy',
        learning_rate=0.001, 
        buffer_size=1000000, 
        learning_starts=100, 
        batch_size=100, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=(1, 'episode'), 
        gradient_steps=- 1, 
        action_noise=None, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        policy_delay=2, 
        target_policy_noise=0.2, 
        target_noise_clip=0.5, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


    dqn = dict(
        policy='MlpPolicy',
        learning_rate=0.0001, 
        buffer_size=1000000, 
        learning_starts=50000, 
        batch_size=32, 
        tau=1.0, 
        gamma=0.99, 
        train_freq=4, 
        gradient_steps=1, 
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        target_update_interval=10000, 
        exploration_fraction=0.1, 
        exploration_initial_eps=1.0, 
        exploration_final_eps=0.05, 
        max_grad_norm=10, 
        tensorboard_log=None, 
        create_eval_env=False, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True,
        policy_kwargs= DEFAULT_POLICY_KWARG),


)
# ----------------------------------------------------------------------------------------------------
