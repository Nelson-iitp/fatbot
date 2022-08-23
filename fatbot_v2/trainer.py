
import fatbot as fb
import torch.nn as nn
import db6 as db


## algorithm
sbalgo = fb.PPO #<----- model args depend on this
sbname = 'PPO'  #<----- model name (save file) depend on this

## parameters (algorithm specific)
total_timestepsL=    [50_000]#, 150_000, 500_000]
schemes =           [db.scheme_1, ]# db.scheme_2, db.scheme_3]
n_steps=            2048+1024
batch_size =        64+32
n_epochs =          10
learning_rate =     0.0003
## networks (algorithm specific)
pkw=dict(
        #log_std_init=-0.2,
        #ortho_init=False,
        activation_fn=nn.LeakyReLU, 
        net_arch=[dict(
            pi=[400, 300, 300], 
            vf=[400, 300, 300])])

delta_reward = True
isd_keys = [db.isd_keys[0]]
permute_states = False
#%%         
if __name__=='__main__':
    # for each delta rew, scheeme, isdkey
    for total_timesteps in total_timestepsL:
        for si,scheme in enumerate(schemes):
            for isd in isd_keys:
                id = f'{sbname}_{si}_{isd}_{total_timesteps}'
                print(f'Training: [{id}]')
                model = sbalgo(policy='MlpPolicy', 
                        env=db.envF(False, scheme, delta_reward, permute_states, isd), 
                        learning_rate = learning_rate,
                        n_steps= n_steps,
                        batch_size = batch_size,
                        n_epochs = n_epochs,
                        gamma = 0.99,
                        gae_lambda= 0.95,
                        clip_range = 0.2,
                        clip_range_vf = None,
                        normalize_advantage= True,
                        ent_coef= 0.005,
                        vf_coef=0.5,
                        max_grad_norm= 0.5,
                        use_sde = False,
                        sde_sample_freq = -1,
                        target_kl = None,
                        verbose=1,
                        seed= None,
                        device='cpu',
                        policy_kwargs=pkw)
                
                model.learn(total_timesteps=total_timesteps,log_interval=5000)
                model.save(id)
        print(f'Finished!')





#%%


"""

model = fb.DDPG(policy='MlpPolicy', env=envF(False), 
        learning_rate = 0.001,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size = 64,
        tau=0.005,
        gamma = 0.999,
        train_freq=(1,"episode"),
        gradient_steps=-1,
        #action_noise=fb.OrnsteinUhlenbeckActionNoise(
        #                mean=np.zeros_like(env.action_space.sample()),
        #                sigma=0.5, theta=0.15, dt=0.01),
        action_noise=fb.NormalActionNoise(
                        mean=np.zeros_like(env.action_space.sample()),
                        sigma=np.ones_like(env.action_space.sample())
        ),
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        create_eval_env=False,
        verbose=1,
        seed= None,
        device='cpu',
        policy_kwargs=dict(
            #log_std_init=-0.2,
            #ortho_init=False,
            activation_fn=nn.LeakyReLU, 
            net_arch=[256,256,256,256])
        )
"""