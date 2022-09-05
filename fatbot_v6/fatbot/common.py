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

def get_nspace(n, dtype, shape, flatten=True, low=None, high=None):
    base_low =  np.zeros(shape=shape, dtype=dtype)  + (0 if (low is None) else (np.array(low)))
    base_high =  np.zeros(shape=shape, dtype=dtype)  + (0 if (high is None) else (np.array(high)))
    low = np.vstack(tuple([ base_low for _ in range(n) ]))
    high = np.vstack(tuple([ base_high for _ in range(n) ]))
    if flatten:
        low = low.flatten()
        high = high.flatten()
    return gym.spaces.Box(
        low= low,
        high= high, 
        shape =low.shape, dtype = dtype) # ((n,) + shape)

def get_angle(P):
    # determin quadrant
    d = np.linalg.norm(P, 2) # np.sqrt(x**2 + y**2)
    #q = 0
    if P[0]>=0:
        if P[1]>=0: #q=1
            t = np.arccos(P[0]/d) # np.arcsin(y/d)
        else: #q=4
            t = 2*np.pi - np.arccos(P[0]/d)
    else:
        if P[1]>=0: #q=2
            t = np.pi - np.arcsin(P[1]/d)
        else: #q=3
            t =  np.pi + (np.arcsin(-P[1]/d))
    return t

def image2video(image_folder, video_name='', fps=1):
    # assume all plots have been saved like: *n.png (*=any, n=0,1,2...)
    # NOTE: after converting to video, we can reduce its size by converting using VLC - File>Convert/Save>  set resoultion=0.5
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
        save_state_info='',
        save_both_states=True,
        ):


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
        if type(starting_state) is str:
            cs = env.custom_reset(starting_state=starting_state) # reset
        else:
            cs = env.reset(starting_state=starting_state) # reset
    
        
        if save_state_info and save_both_states:
            env.save_state(f'{save_state_info}_{episode}_initial.npy')
            fig=env.render() # do a default render
            fig.savefig(f'{save_state_info}_{episode}_initial.png')
            #plt.show()
            del fig


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
        if save_state_info:
            env.save_state(f'{save_state_info}_{episode}_final.npy')
            fig=env.render() # do a default render
            fig.savefig(f'{save_state_info}_{episode}_final.png')
            #plt.show()
            del fig
        #if cast_render:
        #    _=env.render()
        #    plt.show()
        print(f'[x] End Episode: {episode+1}] :: Return: {episode_return}, Steps: {episode_timesteps}')
        
        if (plot_results>1) and (episode_timesteps>1):
            episode_reward_history=np.array(episode_reward_history)
            fig, ax = plt.subplots(2, 1, figsize=(12,6))
            fig.suptitle(f'Episode: {episode+1}')
            ax[0].plot(episode_reward_history[:,0], label='Reward', color='tab:blue')
            ax[1].plot(episode_reward_history[:,1], label='Return', color='tab:green')
            ax[0].legend()
            ax[1].legend()
            #plt.show()
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
        #plt.show()
    return average_return, total_steps


