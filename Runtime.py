# this file is to demo the speed of naive vs optimized

# import time, NumPy & PyOpenCL
import time
import numpy as np
import pyopencl as cl
import pyopencl.array

# import MatPlotLib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import Projection.py, City.py
from Projection import *
from City import NewWorld0


# generate model with various L and record the runtime
time_naive = []
time_optim = []
Lset = [10, 20, 40, 80, 200, 400]

for L in Lset:
    # build city model with different L
    world_cpu = NewWorld0(L).astype(np.int32)

    center      = np.array(world_cpu.shape)/2
    screenshape = (40, 40, 3)
    Projector   = Projection(world_cpu, screenshape)

    phi = 0
    r  = L/2
    px = np.cos(phi)*r
    py = np.sin(phi)*r
    position    = np.array([px, py, r])
    direction   = - position/ np.linalg.norm(position)
    position    = (position+center).astype(np.int32)
    pos_min = [0]*3
    pos_max = [L-1]*3
    position = np.minimum(pos_max,position)
    position = np.maximum(pos_min,position)

    # record runtime
    flag    = 'naive'
    runtime = Projector.proj(position, direction, flag)[1]
    time_naive.append(runtime)
    flag    = 'optimized'
    runtime = Projector.proj(position, direction, flag)[1]
    time_optim.append(runtime)


print('time_naive:')
print(time_naive)
print('time_optim:')
print(time_optim)

# plot runtime
plt.gcf()
plt.plot(Lset, time_naive, 'b', label='naive')
plt.plot(Lset, time_optim, 'r', label='optimized')
plt.legend(loc='upper left')
plt.title('projection algorithm')
plt.xlabel('model size L')
plt.ylabel('run time/sec')
plt.savefig('projection_runtime.png')


