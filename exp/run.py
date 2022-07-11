
#%% [Imports]

import fatbot as fb
import torch.nn as nn


#%% [define environment]


def envF(test):
    from fatbot.db import World_Test as World #<-------------- fatbolt.World class
    return World( # create and return an instance of World
        
        #<-------------- fatbot.Swarm instance (refer fatbot.db.GLOBAL_ISD)
        swarm=fb.db.swarm_4x(False, '4x2', '4x1'),    

        #<-------------- world dynamics 
        horizon=500,                    #<------ set 0 for infinite horizon
        enable_imaging=True,            #<------ enables sensor-ray imaging, disbale if not required
        seed=None,                      #<------ prng seed
        custom_XY=None,                 #<------ custom XY-range for world (by default copies from swarm, change if needed)


        #<-------------- render args 
        record_reward_hist=test,        #<------ if True, records reward history per episode (and renders it as well)
        render_normalized_reward=False, #<------ if True, renders normalized bar plot for reward signal
        render_xray_cmap='hot',         #<------ colormap arg for plt.imshow(xray) 
        render_dray_cmap='copper',      #<------ colormap arg for plt.imshow(dray)
        render_dpi=32,                  #<------ dpi arg for plt.figure()
        render_figure_ratio=0.4,        #<------ figsize multiplier for arg for plt.figure()
        render_bounding_width=0.05,     #<------ how much margin to leave b/w world boundary and figure boundary (%)
            )

#%% [define agent class]

# inherit from fatbot.Agent
class testAgent(fb.Agent):

    def __init__(self, base_dir) -> None:
        model_type='ppo'                #<---- (str) - check fatbot.agents.MODEL_TYPES for available model types
        super().__init__(
            model_type=model_type,
            model_name = f'{str(__class__.__name__)}_{model_type}',
            base_dir=base_dir,
            training_env= envF(False),
            testing_env= envF(True)) 

    def __call__(self): #<--- trains on call, learn_args and model_args are provided to model_type.learn() 
        self.train( #<--- will set self.model to a trained sb3 model

            learn_args = dict(total_timesteps = 10_000, log_interval = 100_000),

            model_args = dict(# env = ? #<----- do not provide env here
                            policy =                    'MlpPolicy',
                            learning_rate=              0.0003, # schedule
                            n_steps=                    2048, 
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

#%% [create agent instance]

agent = testAgent(base_dir='./test_dir') #<----- select base directory (for current run)


#%% [training]

agent() #<---- trains on call

#%% [testing]

average_return, total_steps = \
agent.test(
    episodes=2,                     #<----- total no of episodes
    steps=50,                      #<----- steps per episodes (0 for inifite horizon - i.e. untill env.is_done() returns true)
    deterministic=False,             #<----- arg for model.predict()
    render_mode='all',              #<----- render_mode = all, env, sen, rew  (keep blank for no rendering)
    save_fig='run',                 #<----- (figure directory) for saving plt.figures or (video name) if make_video is true
    save_dpi='figure',              #<----- dpi for plt.savefig(), NOTE: 'figure' means use fig's dpi (as define in env.render_dpi)
    make_video=True,                #<----- if True, compiles a video of all rendered frames
    video_fps=2,                    #<----- frames per second for video (default=1)
    starting_state=None,            #<----- (None :: selects randomly) or (lambda epsisode: state_index :: selects index on given episodes) 
)
print(f'{average_return=}, {total_steps=}')


#%% [done]

print('done!')

