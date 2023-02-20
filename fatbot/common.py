#==============================================================
import numpy as np
import gym, gym.spaces
import os, cv2
from io import BytesIO
from math import inf, pi
import matplotlib.pyplot as plt
import gym, gym.spaces
import datetime
now = datetime.datetime.now
fake = lambda members: type('object', (object,), members)()

class FAKE:
    def __init__(self, **members) -> None: 
        for k,v in members.items(): setattr(self, k, v)

def get_nspace(n, dtype, shape, flatten=True, low=None, high=None):
    r""" Returns a combined space of n similar spaces
        
    ..note:: This is used to create a common shared observation/action space for multiple similar bots

    :param n:       no of spaces to combine
    :param dtype:   dtype of space
    :param shape:   shape of single space
    :param flatten: if True, flattens the shape into 1-D array otherwise stacks using `np.vstack`
    """
    base_low =  np.zeros(shape=shape, dtype=dtype)  + (0 if (low is None) else (np.array(low)))
    base_high =  np.zeros(shape=shape, dtype=dtype)  + (0 if (high is None) else (np.array(high)))
    low = np.vstack(tuple([ base_low for _ in range(n) ]))
    high = np.vstack(tuple([ base_high for _ in range(n) ]))
    if flatten:
        low = low.flatten()
        high = high.flatten()
    return gym.spaces.Box(low= low,high= high, shape =low.shape, dtype = dtype)

def get_espace(n, dtype, shape, edim, low=None, high=None, elow=None, ehigh=None):
    r""" Returns a combined space of n similar spaces and additional dims
        
    ..note:: This is used to create a common shared observation/action space for multiple similar bots

    :param n:       no of spaces to combine
    :param dtype:   dtype of space
    :param shape:   shape of single space
    :param flatten: if True, flattens the shape into 1-D array otherwise stacks using `np.vstack`
    """
    base_low =  np.zeros(shape=shape, dtype=dtype)  + (0 if (low is None) else (np.array(low)))
    base_high =  np.zeros(shape=shape, dtype=dtype)  + (0 if (high is None) else (np.array(high)))
    low = np.vstack(tuple([ base_low for _ in range(n) ])).flatten()
    high = np.vstack(tuple([ base_high for _ in range(n) ])).flatten()
    ext_low =  np.zeros(shape=(edim,), dtype=dtype)  + (0 if (elow is None) else (np.array(elow)))
    ext_high =  np.zeros(shape=(edim,), dtype=dtype)  + (0 if (ehigh is None) else (np.array(ehigh)))
    low = np.hstack((low, ext_low))
    high = np.hstack((high, ext_high))
    return gym.spaces.Box(low= low,high= high, shape =low.shape, dtype = dtype)

def get_angle(P):
    r""" Helper function - used to determine angle (in rads) 
        made by a position vector P with the +ve x-axis """
    # determin quadrant (q)
    d = np.linalg.norm(P, 2) # np.sqrt(x**2 + y**2)
    if P[0]>=0: 
        if P[1]>=0: # 1st quadrant
            t = np.arccos(P[0]/d) # or np.arcsin(y/d)
        else: # 4th quadrant
            t = 2*np.pi - np.arccos(P[0]/d)
    else:
        if P[1]>=0: # 2nd quadrant
            t = np.pi - np.arcsin(P[1]/d)
        else: # third quadrant
            t =  np.pi + (np.arcsin(-P[1]/d))
    return t

def image2video(image_folder, video_name='', fps=1):
    r""" Converts a set of images to a video and saves it at desired path, 
        assume all plots have been saved like: *n.png (*=any, n=0,1,2...)
    
    :param video_name:  name of the video file, if left blank, uses the image_folder name
    ..note:: after converting to video, we can reduce its size by converting using VLC - File>Convert/Save>  set resoultion=0.5
    """
    import cv2, os
    file_list = []
    for f in os.listdir(image_folder):
        if f.lower().endswith('.png'):
            file_list.append(f)
    if not file_list:
        print(f'Image Folder is Empty!')
        return

    video_path = os.path.join(os.path.dirname(image_folder), 
        (f'{video_name}.avi' if video_name else  f'{os.path.basename(image_folder)}.avi'))
    x = np.array(file_list)
    y = np.argsort(np.array([ int(i.split(".")[0]) for i in x ]).astype(np.int64))
    images = x[y]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_path, 0, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

class RandomPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.action_space.sample(), None

class ZeroPolicy:
    def __init__(self, action_space) -> None:
        self.action_space = action_space
        self.zero = self.action_space.sample()*0.0
    def predict(self, observation, **kwargs): # state=None, episode_start=None, deterministic=True
        return self.zero, None
    
class REMAP:
    def __init__(self,Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm

class JSON:
    import json # load only when called
    def save(path, data_dict):
        """ saves a dict to disk in json format """
        with open(path, 'w') as f:
            f.write(__class__.json.dumps(data_dict, sort_keys=False, indent=4))
        return path
    def load(path):
        """ returns a dict from a json file """
        data_dict = None
        with open(path, 'r') as f:
            data_dict = __class__.json.loads(f.read())
        return data_dict

class RenderHandler:

    def __init__(self, env, render_as='', save_dpi='figure', make_video=False, video_fps=1, render_kwargs={}, start_n=0) -> None:
        self.env=env #<--- base env that returns a plt.Figure when render is called
        self.render_kwargs = render_kwargs #<---- a key mode from prebuilt render_modes
        self.render_as=render_as
        self.save_dpi = save_dpi
        self.make_video=make_video
        self.video_fps=video_fps
        self.start_n=start_n
        
        # make render functions
        self.Start = self.noop
        self.Render = self.noop
        self.Stop = self.noop
        
        if render_as is not None:
            if render_as: # not blank
                if make_video:
                    self.Start = self.Start_Video
                    self.Render = self.Render_Video
                    self.Stop = self.Stop_Video
                else:
                    self.Start = self.Start_Image
                    self.Render = self.Render_Image
            else:
                self.Render = self.Render_Show

    def noop(self):
        pass

    def Render_Show(self):
        fig = self.env.render(**self.render_kwargs)
        plt.show()
        del fig

    def Start_Image(self):
        os.makedirs(self.render_as, exist_ok=True)
        self.n=self.start_n

    def Render_Image(self):
        fig = self.env.render(**self.render_kwargs)
        fig.savefig(os.path.join(self.render_as, f'{self.n}.png'), dpi=self.save_dpi, transparent=False )     
        plt.close()
        self.n+=1
        del fig

    def Start_Video(self):
        # video handler requires 1 env.render to get the shape of env-render figure
        self.buffer = BytesIO()

        #self.buffer.seek(0) # seek zero before writing - not required on first write
        self.env.sample_render_fig(**self.render_kwargs).savefig( self.buffer, dpi=self.save_dpi, transparent=False ) 
        self.buffer.seek(0) # seek zero before reading
        frame = cv2.imdecode(np.asarray(bytearray(self.buffer.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.height, self.width, _ = frame.shape
        self.video_file_name = self.render_as+'.avi'

        #                                   file,     fourcc, fps, size
        self.video = cv2.VideoWriter(self.video_file_name , 0, self.video_fps, (self.width, self.height)) 

        # self.video.write(frame) #<--- do not write yet
        print(f'[{__class__.__name__}]:: Started Video @ [{self.video_file_name}] :: Size [{self.width} x {self.height}]')

    def Render_Video(self):
        self.buffer.seek(0) # seek zero before writing 
        fig = self.env.render(**self.render_kwargs)
        fig.savefig( self.buffer, dpi=self.save_dpi, transparent=False ) 
        plt.close()
        del fig
        self.buffer.seek(0) # seek zero before reading
        self.video.write(cv2.imdecode(np.asarray(bytearray(self.buffer.read()), dtype=np.uint8), cv2.IMREAD_COLOR))

    def Stop_Video(self):
        cv2.destroyAllWindows()
        self.video.release()
        self.buffer.close()
        del self.buffer
        print(f'[{__class__.__name__}]:: Stopped Video @ [{self.video_file_name}]')

def TEST(
        env, 
        model=None, 
        episodes=1, 
        steps=0, 
        deterministic=True, 
        render_as='', 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs={},
        starting_state=None,
        plot_results=0,
        start_n=0,
        reverb=0,
        plot_end_states=False,
        save_states='',
        save_prefix=''
        ):
    if save_states:
        os.makedirs(save_states, exist_ok=True)
    sehist=[]
    tehist=[]
    fighisti=[]
    fighistf=[]
    renderer = RenderHandler(env, render_as=render_as, save_dpi=save_dpi, make_video=make_video,
                                            video_fps=video_fps, render_kwargs=render_kwargs, start_n=start_n )
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
        if plot_end_states: fighisti.append(env.render(*render_kwargs))
        if save_states:
            env.save_state(os.path.join(save_states, f'{episode}_{save_prefix}_initial.npy'))
        print(f'\n[+] Begin Episode: {episode+1} of {episodes}')
        

        episode_return = 0.0
        episode_timesteps = 0
        episode_reward_history = []
        max_return = -np.inf
        max_return_at = -1
        

        renderer.Render()  #<--- open renderer and do 1st render
        while (not done) and (episode_timesteps<episode_max_steps):
            action, _ = model.predict(cs, deterministic=deterministic) # action = env.action_space.sample() #print(action)
            cs, rew, done , _ = env.step(action)
            episode_return += rew
            episode_reward_history.append((rew, episode_return))
            if episode_return>max_return:
                max_return=episode_return
                max_return_at=episode_timesteps

            episode_timesteps+=1
            if reverb: print(f'  [{episode_timesteps}/{done}]: Reward: {rew}')
            renderer.Render() 

        if plot_end_states:fighistf.append(env.render(*render_kwargs))
            #plt.show()
        if save_states:
            env.save_state(os.path.join(save_states, f'{episode}_{save_prefix}_final.npy'))
        print(f'[x] End Episode: {episode+1}] :: Return: {episode_return}, Steps: {episode_timesteps}')
        sehist.append( max_return_at )
        tehist.append(episode_reward_history)
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
    return average_return, total_steps, sehist, tehist, fighisti, fighistf

def TEST2(
        env, 
        model1=None, 
        model2=None,
        episodes=1, 
        steps=0, 
        deterministic=True, 
        render_as='', 
        save_dpi='figure', 
        make_video=False,
        video_fps=1,
        render_kwargs={},
        starting_state=None,
        plot_results=0,
        start_n=0,
        reverb=0,
        plot_end_states=False,
        save_states='',
        save_prefix='',
        last_n_steps=(20,20), last_deltas=(0.005, 0.005),
        initial_steps=0,
        ):
    # for testing using two models
    if save_states:
        os.makedirs(save_states, exist_ok=True)
    sehist=[]
    tehist=[]
    fighisti=[]
    fighistf=[]
    renderer = RenderHandler(env, render_as=render_as, save_dpi=save_dpi, make_video=make_video,
                                            video_fps=video_fps, render_kwargs=render_kwargs, start_n=start_n )
    episode_max_steps = (steps if steps>0 else inf)
    print(f'[.] Testing for [{episodes}] episodes @ [{episode_max_steps}] steps')
    if model1 is None:
        print('[!] No model1 provided - Using random actions')
        model1 = RandomPolicy(env.action_space)
    if model2 is None:
        print('[!] No model2 provided - Using random actions')
        model2 = RandomPolicy(env.action_space)
    # start episodes
    renderer.Start()
    episodes = (episodes if episodes>1 else 1)
    test_history = []
    print(f'\n[++] Begin Epoch: Running for {episodes} episodes')
    #last_n_step=20, last_delta=0.005
    for episode in range(episodes):
        cs = env.reset(starting_state=starting_state) # reset
        done = False
        if plot_end_states:fighisti.append(env.render(*render_kwargs))
        if save_states:
            env.save_state(os.path.join(save_states, f'{episode}_{save_prefix}_initial.npy'))
        print(f'\n[+] Begin Episode: {episode+1} of {episodes}')
        

        episode_return = 0.0
        episode_timesteps = 0
        episode_reward_history = []
        max_return = -np.inf
        max_return_at = -1
                
        # r=rew
        # l=len(xxx)
        # n=20
        # delta_avg=0.0005
        # avg = []
        # for i in range(n, l):
        # avg.append(np.mean(r[i-n:i]))
        # avg = np.array(avg)
        # plt.figure(figsize=(18,6))
        # plt.plot(avg)
        # print('len:', len(avg),'/', l)

        # ix = np.where(abs(avg)<=delta_avg)[0]
        # print(ix)
        # plt.scatter(ix, avg[ix])

        renderer.Render()  #<--- open renderer and do 1st render
        model = model1
        arew = []
        switch_model=True
        eps_decay = 1.0
        last_n_step = last_n_steps[0]
        last_delta = last_deltas[0]
        decay_by = 1/(last_n_steps[1]*2)
        zstate = env.action_space.sample()*0
        while (not done) and (episode_timesteps<initial_steps):
            episode_timesteps+=1
            env.step(zstate)
            renderer.Render() 
        episode_timesteps=0
        while (not done) and (episode_timesteps<episode_max_steps):
            action, _ = model.predict(cs, deterministic=deterministic) # action = env.action_space.sample() #print(action)
            if not switch_model:
                action*=eps_decay
                eps_decay=max(0.0, eps_decay-decay_by)
            cs, rew, done , _ = env.step(action)
            episode_return += rew
            episode_reward_history.append((rew, episode_return))
            arew.append(rew)
            if episode_return>max_return:
                max_return=episode_return
                max_return_at=episode_timesteps

            episode_timesteps+=1
            if reverb: print(f'  [{episode_timesteps}/{done}]: Reward: {rew}')
            if len(arew)>=last_n_step:
                avgrew = np.average(arew)
                del(arew[0])

                if abs(avgrew)<=last_delta:
                    
                    if switch_model:
                        print('Switch model')
                        model=model2
                        switch_model=False
                        arew.clear()
                        last_n_step = last_n_steps[1]
                        last_delta = last_deltas[1]
                    else:
                        print('Early Stop')
                        done = True



            renderer.Render() 

        if plot_end_states: fighistf.append(env.render(*render_kwargs))
        if save_states:
            env.save_state(os.path.join(save_states, f'{episode}_{save_prefix}_final.npy'))
        print(f'[x] End Episode: {episode+1}] :: Return: {episode_return}, Steps: {episode_timesteps}')
        sehist.append( max_return_at )
        tehist.append(episode_reward_history)
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
    return average_return, total_steps, sehist, tehist, fighisti, fighistf



def log_evaluations(evaluations_path):
    E = np.load(evaluations_path)
    # E.files # ['timesteps', 'results', 'ep_lengths']
    ts, res, epl = E['timesteps'], E['results'], E['ep_lengths']
    # ts.shape, res.shape, epl.shape #<---- (eval-freq, n_eval_episodes)
    resM, eplM = np.mean(res, axis=1), np.mean(epl, axis=1) # mean reward of all eval_episodes

    fr=plt.figure(figsize=(8,4))
    plt.scatter(ts, resM, color='green')
    plt.plot(ts, resM, label='mean_val_reward', color='green')
    plt.xlabel('step')
    plt.ylabel('mean_val_reward')
    plt.legend()
    #plt.show()
    #plt.close()


    fs=plt.figure(figsize=(8,4))
    plt.scatter(ts, eplM, color='blue')
    plt.plot(ts, eplM, label='mean_episode_len', color='blue')
    plt.xlabel('step')
    plt.ylabel('mean_episode_len')
    plt.legend()
    #plt.show()
    #plt.close()

    E.close()
    return fr, fs