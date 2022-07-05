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
    assert(d>0)
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