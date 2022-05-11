import numpy as np
import random
from scipy.interpolate import interp1d

def resample(x,M,N):
    timesteps = x.shape[1]
    orig_steps = np.arange(timesteps)
    interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/(M+1))
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)

    length_inserted = InterpVal.shape[1]
    start = random.randint(0, length_inserted - timesteps * (N + 1))
    index_selected = np.arange(start, start + timesteps * (N + 1), N + 1)
    return InterpVal[:, index_selected, :]
def resample_random(x):
    M,N = random.choice([[1,0],[2,1],[3,2]])
    timesetps = x.shape[1]
    orig_steps = np.arange(timesetps)
    interp_steps = np.arange(0, orig_steps[-1]+0.001, 1/(M+1))
    Interp = interp1d(orig_steps, x, axis=1)
    InterpVal = Interp(interp_steps)

    length_inserted = InterpVal.shape[1]
    start = random.randint(0, length_inserted - timesetps * (N + 1))
    index_selected = np.arange(start, start + timesetps * (N + 1), N + 1)
    return InterpVal[:, index_selected, :]