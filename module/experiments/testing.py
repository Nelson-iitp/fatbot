#==============================================================
import numpy as np

import matplotlib.pyplot as plt
from fatbot.common import TEST, TEST2, TEST3, pjs, pj, ZeroPolicy, RandomPolicy


global_render_kwargs=dict(
    local_sensors=False, 
    reward_signal=True, 
    fill_bots=True, 
    show_com=2,
    state_hist_marker='o')

global_aux_dampening = True
initial_waiting_steps = 10
final_waiting_steps =   10

def do_checking(env, episodes, steps=1, use_random_actions=False, verbose=0, save_prefix='', save_path='', save_images=True):
    average_return, avg_steps, sehist, tehist, test_history, fi, ff = TEST(
        env=env, 
        model= RandomPolicy(env.action_space) if use_random_actions else ZeroPolicy(env.action_space), 
        episodes=episodes, 
        steps=steps, 
        deterministic=True, 
        render_as=None, 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs=global_render_kwargs,
        starting_state=None,

        start_n=0,
        reverb=verbose,
        plot_end_states=save_images,
        save_states='', #<----------- env conf
        save_prefix=''
    )
    print(f'\n{average_return=}\n{avg_steps=}\n')    
    if save_images:
        assert(len(sehist)==len(tehist))
        for i,f in enumerate(fi): f.savefig(pjs(save_path, f'{save_prefix}_{i+1}_begin.png'))
        for i,f in enumerate(ff): f.savefig(pjs(save_path,f'{save_prefix}_{i+1}_end.png'))

def do_testing_(env, model, episodes, model_type, verbose=0, save_states='', save_images=False):
    average_return, avg_steps, sehist, tehist, test_history, fi, ff = TEST(
        env=env, 
        model=model, 
        episodes=episodes, 
        steps=0, 
        deterministic=True, 
        render_as=None, 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs=global_render_kwargs,
        starting_state=None,

        start_n=0,
        reverb=verbose,
        plot_end_states=save_images,
        save_states=save_states, #<----------- env conf
        save_prefix=model_type
    )
    print(f'\n{average_return=}\n{avg_steps=}\n')    
    if save_images:
        assert(len(sehist)==len(tehist))
        for i,f in enumerate(fi): f.savefig(pjs(save_states, f'{model_type}_{i+1}_A.png'))
        for i,f in enumerate(ff): f.savefig(pjs(save_states, f'{model_type}_{i+1}_Z.png'))

        fig, ax = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(f'Rewards @ Episodes {len(tehist)}')
        ax[0].set_ylabel('Reward')
        ax[1].set_ylabel('Return')
        for episode,episode_reward_history in enumerate(tehist):
            max_reward_at = sehist[episode]
            episode_reward_history=np.array(episode_reward_history)
            #fig.suptitle(f'Episode: {episode+1}')
            rewh, reth = episode_reward_history[:,0], episode_reward_history[:,1]
            ax[0].plot(rewh, label='Reward', color='tab:blue', linewidth=0.6)
            ax[0].scatter([max_reward_at],rewh[max_reward_at], color='tab:blue')
            ax[1].plot(reth, label='Return', color='tab:green', linewidth=0.6)
            ax[1].scatter([max_reward_at],reth[max_reward_at], color='tab:green')

        fig.savefig(pjs(save_states, f'{model_type}_test_per_episode.png'))
        plt.close()

        fig, ax = plt.subplots(2, 1, figsize=(12,6))
        fig.suptitle(f'Test Results')
        ax[0].plot(test_history[:,0], label='Steps', color='tab:purple')
        ax[1].plot(test_history[:,1], label='Return', color='tab:green')
        ax[0].legend()
        ax[1].legend()
        fig.savefig(pjs(save_states, f'{model_type}_test_all_episodes.png'))
        plt.close()

def do_testing(model_type, testing_env, eval_path, model_algo, episodes, verbose=0, save_states='', save_images=False):
    model_path=pjs(eval_path, model_type)
    model = model_algo.load(model_path)
    print(f'Testing @ [{model_path}] for {episodes=}')
    do_testing_(testing_env, model, episodes, model_type, verbose, save_states, save_images)

def do_testing2(env, model1, model2,  verbose=0, episodes=1, 
                last_n_steps=(20,20), last_deltas=(0.005, 0.005),):
    #average_return, avg_steps, sehist, tehist, test_history, fi, ff, switch_history
    average_return, avg_steps, sehist, tehist, test_history, fi, ff, switch_history = TEST2(
        env=env, 
        model1=model1, 
        model2=model2,
        episodes=episodes, 
        steps=0, 
        deterministic=True, 
        render_as=None, 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs=global_render_kwargs,
        starting_state=None,
        start_n=0,
        reverb=verbose,
        plot_end_states=False,
        last_n_steps=last_n_steps, last_deltas=last_deltas,
        save_states='', #<----------- env conf
        save_prefix='',
        initial_steps=0,
    )
    print(f'\n{average_return=}\n{avg_steps=}\n')    
    return average_return, avg_steps, sehist, tehist, test_history, fi, ff, switch_history

def do_testing3(env, model1, model2, model_type, 
                make_video=False, verbose=0, save_states='', save_images=False, render_as=None, last_n_steps=(20,20), last_deltas=(0.005, 0.005),):
    average_return, avg_steps, sehist, tehist, test_history, fi, ff, fs, switch_history = TEST3(
        env=env, 
        model1=model1, 
        model2=model2,
        episodes=1, 
        steps=0, 
        deterministic=True, 
        render_as=render_as, 
        save_dpi='figure', 
        make_video=make_video,
        video_fps=1,
        render_kwargs=global_render_kwargs,
        starting_state=None,
        start_n=0,
        reverb=verbose,
        plot_end_states=save_images,
        last_n_steps=last_n_steps, last_deltas=last_deltas,
        save_states='', #<----------- env conf
        save_prefix='',
        initial_steps=initial_waiting_steps,
        final_steps=final_waiting_steps,
        decay_2=global_aux_dampening,
    )
    print(f'\n{average_return=}\n{avg_steps=}\n')    
    np.savez(pjs(save_states, f'{model_type}_results.npz'),
        sh = np.array(sehist), th=np.array(tehist), wh=np.array(switch_history))
    if save_images:
        assert(len(sehist)==len(tehist))
        for i,f in enumerate(fi): f.savefig(pjs(save_states, f'{model_type}_{i+1}_A.png'), dpi=100)
        for i,f in enumerate(fs): f.savefig(pjs(save_states, f'{model_type}_{i+1}_S.png'), dpi=100)
        for i,f in enumerate(ff): f.savefig(pjs(save_states, f'{model_type}_{i+1}_Z.png'), dpi=100)

