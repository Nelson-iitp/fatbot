GLOBAL_SYMBOLS = """ α β γ δ ε ζ η θ κ λ μ ξ π ρ ς σ φ ψ ω τ ϐ ϑ ϕ Ω ℓ Λ Γ Θ ϴ Φ Ψ Ξ Δ """
import datetime
import numpy as np
import gym.spaces
now = datetime.datetime.now
fake = lambda members: type('object', (object,), members)()

def get_nspace(n, dtype, shape, low=None, high=None):
    base_low =  np.zeros(shape=shape, dtype=dtype)  + (0 if (low is None) else (np.array(low)))
    base_high =  np.zeros(shape=shape, dtype=dtype)  + (0 if (high is None) else (np.array(high)))
    low = np.vstack(tuple([ base_low for _ in range(n) ])).flatten()
    high= np.vstack(tuple([ base_high for _ in range(n) ])).flatten()
    return gym.spaces.Box(low= low,high= high, shape =low.shape, dtype = dtype) # ((n,) + shape)

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

def image2video(image_folder, video_name=''):
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
    video = cv2.VideoWriter(video_path, 0, 1, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()






"""ARCHIVE

def get_hspace(n, dtype, shapeL, lowL=None, highL=None):
    # here shape, low, high are all lists
    h_shape = ()
    h_low, h_high = [], []
    for shape,low,high in zip(shapeL, lowL, highL):
        h_shape = h_shape + shape
        h_high.append   (np.zeros(shape=shape, dtype=dtype)  + (0 if (high is None) else (np.array(high))))
        h_low.append    (np.zeros(shape=shape, dtype=dtype)  + (0 if (low is None)  else (np.array(low))))

    base_low = np.hstack(tuple(h_low))
    base_high = np.hstack(tuple(h_high))
    low = np.vstack(tuple([ base_low for _ in range(n) ])).flatten()
    high= np.vstack(tuple([ base_high for _ in range(n) ])).flatten()
    return gym.spaces.Box(low= low,high= high, shape =low.shape, dtype = dtype) # ((n,) + shape)

"""