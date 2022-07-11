
#%% [Imports]

import fatbot as fb

#%% [define environment]


def envF(testing):
    from fatbot.db import World_Default as World #<-------------- fatbolt.World class
    return World( # create and return an instance of World
        
        #<-------------- fatbot.Swarm instance (refer fatbot.db.GLOBAL_ISD)
        swarm=fb.db.swarm_4x(False, '4x2', '4x1'),    

        #<-------------- world dynamics 
        horizon=500,                    #<------ set 0 for infinite horizon
        enable_imaging=True,            #<------ enables sensor-ray imaging, disbale if not required
        seed=None,                      #<------ prng seed
        custom_XY=None,                 #<------ custom XY-range for world (by default copies from swarm, change if needed)
        delta_reward = False,            #<------ if True, Uses delta (deifference) of rewards (reward improvement)

        #<-------------- render args 
        record_reward_hist=testing,        #<------ if True, records reward history per episode (and renders it as well)
        render_normalized_reward=False, #<------ if True, renders normalized bar plot for reward signal
        render_xray_cmap='hot',         #<------ colormap arg for plt.imshow(xray) 
        render_dray_cmap='copper',      #<------ colormap arg for plt.imshow(dray)
        render_dpi=32,                  #<------ dpi arg for plt.figure()
        render_figure_ratio=0.4,        #<------ figsize multiplier for plt.figure()
        render_bounding_width=0.05,     #<------ how much margin to leave b/w world boundary and figure boundary (%)
            )

#%% [define agent]

agent = fb.Agent(
        model_type='ppo', 
        model_name='MyName',
        model_args=None, # keep None for default args
        base_dir='./MyBaseDir', 
        training_env=envF(False), 
        testing_env=envF(True)
        )


#%% [training]

agent.train(total_timesteps=15_00) #<---- trains on call

#%% [testing]

average_return, total_steps = \
agent.test(
    episodes=1,                     #<----- total no of episodes
    steps=200,                      #<----- steps per episodes (0 for inifite horizon - actual horizon = min(steps, env.horizon)
    deterministic=False,             #<----- arg for model.predict()
    render_mode='all',              #<----- render_mode = all, env, sen, rew  (keep blank for no rendering)
    save_as='MyOutput',      #<----- (figure directory) for saving plt.figures or (video name) if make_video is true
    save_dpi='figure',              #<----- dpi for plt.savefig(), NOTE: 'figure' means use fig's dpi (as define in env.render_dpi)
    make_video=True,                #<----- if True, compiles a video of all rendered frames
    video_fps=2,                    #<----- frames per second for video (default=1)
    starting_state=None,            #<----- (None :: selects randomly) or (lambda epsisode: state_index :: selects index on given episodes)
    plot_results=0                  #<----- 0=no plot, 1=plot end result only, 2=plot per episode result and end result
)
print(f'{average_return=}, {total_steps=}')


#%% [done]

print('done!')

