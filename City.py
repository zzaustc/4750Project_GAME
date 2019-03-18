# build city, buildings
# save the model as Model.npy


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



##########################
# build a building
##########################

# make sure every points in boundary
def inboundary(point, worldsize):
    for i in range(3):
        point[i] = point[i] if point[i]<worldsize[i] else worldsize[i]
    return point

# draw a cuboid with color
def cuboid(world, point1, point2, color):
    [x1, y1, z1] = point1
    [x2, y2, z2] = point2
    world[x1:x2, y1:y2, z1:z2] = color
    return world

# draw a border for the cuboid
def cuboidframe(world, point1, point2, color, linewidth):
    # based on cuboid
    [lx, ly, lz] = point2-point1
    lw = linewidth

    # 4 lines parallel to plane XOY
    # 2 lines parallel to axis X
    p1List = [
    point1+(0, 0, lz-lw),
    point1+(0, ly-lw, lz-lw)
    ]
    for p1 in p1List:
        p2 = p1+(lx, lw, lw)
        world = cuboid(world, p1, p2, color)
    # 2 lines parallel to axis Y
    p1List = [
    point1+(0, 0, lz-lw),
    point1+(lx-lw, 0, lz-lw)
    ]
    for p1 in p1List:
        p2 = p1+(lw, ly, lw)
        world = cuboid(world, p1, p2, color)

    # 4 lines parallel to axis Z
    p1List = [
    point1,
    point1+(0, ly-lw, 0),
    point1+(lx-lw, 0, 0),
    point1+(lx-lw, ly-lw, 0)
    ]
    for p1 in p1List:
        p2 = p1+(lw, lw, lz)
        world = cuboid(world, p1, p2, color)
    return world

# draw a new building
def NewBuilding(world, point1, point2, color):
    # based on inboundary, cuboid, cuboidframe
    worldsize = world.shape
    point1 = inboundary(point1, worldsize)
    point2 = inboundary(point2, worldsize)
    world = cuboid(world, point1, point2, color)
    world = cuboidframe(world, point1, point2, ColorLine, LINEWIDTH)
    return world

# draw a cylinder (as the base of city)
def CircleBase(world, z0, color):
    worldsize = world.shape
    Lx, Ly, Lz = worldsize
    r = int(Lx/2)
    for x in range(-r, r):
        for y in range(-r, r):
            if x*x+y*y<r*r:
                world[x+r, y+r, 0:z0] = color
    return world

# draw many buildings
def City(world, color):
    worldsize = world.shape
    Lx, Ly, Lz = worldsize
    r = Lx/2
    center = np.array([r, r, 0])
    # sin45 = 0.85
    LowBound = -0.6*r
    UpperBound = 0.6*r
    grid = 6
    gridsize = (UpperBound-LowBound)/grid
    for x1 in np.arange(LowBound, UpperBound, gridsize):
        for y1 in np.arange(LowBound, UpperBound, gridsize):
            rand3 = np.random.uniform(0, 1, 3)
            deta = (0.4+0.5*rand3)*gridsize
            height = 0.6*np.exp(-4*(x1*x1+y1*y1)/np.power(r,2))*Lz
            point1 = np.ceil(np.array([x1, y1, 0])+center).astype(np.int)
            point2 = np.ceil(np.array([x1, y1, height])+center+deta).astype(np.int)
            # print(point1, point2, height)
            world = NewBuilding(world, point1, point2, color)
    return world


##########################
# visual tools
##########################

# print a slice of the world matrix
def SliceWorld(world, z0=0):
    [Lx, Ly, Lz] = world.shape
    Z = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            Z[i][j] = world[i][j][z0]
    print(Z)


# plot the 3D world
def ViewWorld(world):
    [Lx, Ly, Lz] = world.shape
    X = np.arange(0, Lx)
    Y = np.arange(0, Ly)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((Lx, Ly))
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz-1, -1, -1):
                if world[i][j][k]:
                    Z[i][j] = k
                    break
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
    plt.show()


##########################
# function for building a city
##########################

# constant parameters
ColorAir        = 0
ColorLine       = 1
ColorGround     = 2
ColorBuilding   = 3
LINEWIDTH       = 2

# return world, type: 3d nparray
def NewWorld0(L=40):
    worldsize = (L, L, L)
    [Lx, Ly, Lz] = worldsize
    world = np.zeros(worldsize)
    world = cuboid(world, (0, 0, 0), (Lx, Ly, 2), ColorGround)
    point1 = np.ceil((Lx*0.3, Ly*0.3, 0)).astype(np.int32)
    point2 = np.ceil((Lx*0.7, Ly*0.7, Lz*0.7)).astype(np.int32)
    world = NewBuilding(world, point1, point2, ColorBuilding)
    return world


def NewWorld1(L=200):
    worldsize = (L, L, L)
    [Lx, Ly, Lz] = worldsize
    world = np.zeros(worldsize)
    world = City(world, ColorBuilding)
    world = CircleBase(world, 3, ColorGround)
    return world


##########################
# main
##########################

if __name__ == '__main__':
    # L is the world size, by default 200
    # recommended: set L from 10 to 200
    L0 = 40
    L1 = 200

    # build the world model
    world0 = NewWorld0(L0)
    world1 = NewWorld1(L1)

    # the view function can only be used offline
    #ViewWorld(world0)
    #ViewWorld(world1)

    np.save('MyCityModel0', world0)
    np.save('MyCityModel1', world1)

