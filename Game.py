# this file is for game playing

# import time, NumPy & PyOpenCL
import time
import numpy as np
import pyopencl as cl
import pyopencl.array

# import MatPlotLib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# import Projection.py
from Projection import *


##########################
# changable parameters
##########################
# screenshape = (N, N, 3), resolution of projection image
# recommended: set N from 40 to 400
#screenshape = (40, 40, 3)
#screenshape = (100, 100, 3)
screenshape = (400, 400, 3)

# flag = 'naive' or 'optimized' or 'colorful', projection mode
#flag  = 'naive'
#flag  = 'optimized'
flag  = 'colorful'

# load one city model
#world_cpu   = np.load('MyCityModel0.npy').astype(np.int32)
world_cpu   = np.load('MyCityModel1.npy').astype(np.int32)




##########################
# Now Play !
##########################
# trace1: helix
# trace2: fly over
# trace3: go through

# initialize
Projector   = Projection(world_cpu, screenshape)
Lx, Ly, Lz  = world_cpu.shape
center      = np.array(world_cpu.shape)/2
r      = Lx/2
step   = 6
Trace1 = []
Trace2 = []
Trace3 = []

phi = 0.1
while phi<=np.pi:
    px = np.cos(phi)*r
    py = np.sin(phi)*r
    pz = (phi/np.pi - 0.5)*2*r
    Trace1.append([px, py, pz])
    phi += np.pi/step

phi = 0
while phi<=np.pi:
    py = np.cos(phi)*r
    pz = np.sin(phi)*r
    Trace2.append([0, py, pz])
    phi += np.pi/step

phi = 0
while phi<np.pi:
    py = (phi/np.pi - 0.5)*2*r
    Trace3.append([0, py, 0])
    phi += np.pi/step

Trace = np.array(Trace1)
# Trace = np.array(Trace1 + Trace2 + Trace3)

for position in Trace:
    direction   = - position/ np.linalg.norm(position)
    position    = (position+center).astype(np.int32)
    pos_min = [0]*3
    pos_max = [Lx-1]*3
    position = np.minimum(pos_max,position)
    position = np.maximum(pos_min,position)

    img3d = Projector.proj(position, direction, flag)[0]

    # 0 for debug, 1 for plot, keep is_RGB = 1
    is_RGB = 1
    if is_RGB:
        # plot projection image
        a = plt.axes([0, 0, 1, 1])
        a.imshow(img3d)
        plt.xticks([])
        plt.yticks([])

        # miniplot
        phi1 = np.arange(0, np.pi*2, 0.01)
        x1 = np.cos(phi1)*r
        y1 = np.sin(phi1)*r
        x2 = position[0]-r
        y2 = position[1]-r
        a = plt.axes([.85, .85, .15, .15])
        plt.plot(x1, y1, 'k', x2, y2, 'ks')
        plt.xlabel('[x,y] position')
        plt.xticks([])
        plt.yticks([])

        # save
        plt.savefig('view.png')
        plt.clf()
    else:
        img0 = img3d[:,:,0]
        print(img0)
    time.sleep(1)

