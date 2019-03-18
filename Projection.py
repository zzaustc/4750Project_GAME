import numpy as np
import pyopencl as cl
import pyopencl.array

"""
input:    3D world
output:   2D screen
bitmap figure, all int32
"""
class Projection:

    def __init__(self, world_cpu, screenshape):
        # init
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        # Set up a command queue:
        ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # parameter
        self.worldshape  = world_cpu.shape
        self.screenshape = screenshape


        # COLOR
        COLOR_cpu = np.array([
            # from bright to dark
            [102,255,255], [  0,  0,153], # air
            [  0,  0,  0], [  0,  0,  0], # black solid line
            [204,255, 51], [102,153,  0], # ground
            [255,102,  0], [255,255,  0], # building
            ]).astype(np.int32)
        self.COLOR_gpu = cl.array.to_device(self.queue, COLOR_cpu)

        # device
        screen_cpu          = np.zeros(self.screenshape, dtype=np.int32)
        worldshape_cpu      = np.array(self.worldshape,  dtype=np.int32)
        self.world_gpu      = cl.array.to_device(self.queue, world_cpu)
        self.screen_gpu     = cl.array.to_device(self.queue, screen_cpu)
        self.worldshape_gpu = cl.array.to_device(self.queue, worldshape_cpu)

        # kernel
        kernel_code = """
        // index i, j     for index on the screen
        // index x, y, z  for index of 3D city model


        __kernel void proj_naive(\
            __global int*   world,    __global int*   worldshape,\
            __global int*   screen,   __global int*   COLOR,     \
            __global int*   position, __global float* direction, \
            __global float* i_dir,    __global float* j_dir)
        {
            // 0 for test, 1 for plot
            int is_RGB = 1;
            // position on screen
            int i0 = get_global_id(0);
            int j0 = get_global_id(1);
            int Si = get_global_size(0);
            int Sj = get_global_size(1);

            // load parameters
            int Lx = worldshape[0], Ly = worldshape[1], Lz = worldshape[2];

            /***********************************/
            // ray tracer
            /***********************************/

            // get ray_direction
            int   ray[3];
            float ray_dir[3];
            float view = 1.5;
            for (int id = 0; id < 3; ++id)
                ray_dir[id] = direction[id]+ \
                              view*(i0-0.5*Si)/Sj * i_dir[id] + view*(j0-0.5*Sj)/Sj * j_dir[id];

            // query each pixel on the ray
            // T = sqrt(Lx*Lx+Ly*Ly+Lz*Lz), T is 1.71 Lx
            int T = 1.71*Lx;
            bool is_out = 0;
            for (int t = 0; t < T; ++t)
            {
                for (int id = 0; id < 3; ++id)
                {
                    ray[id] = position[id]+round(t*ray_dir[id]);
                    if (ray[id]<0 || ray[id]>=worldshape[id]) is_out = 1;
                }
                if (is_out){
                    // set background color to air
                    if (is_RGB)
                        for (int id = 0; id < 3; ++id)
                            screen[(i0*Sj + j0)*3 + id] = COLOR[id];
                    else screen[(i0*Sj + j0)*3] = 0;
                    return;
                }

                //  pixel = world(ray[0], ray[1], ray[2])
                int pixel = world[ray[0]*Ly*Lz + ray[1]*Lz +ray[2]];
                if (pixel)
                {
                    if (is_RGB)
                        for (int id = 0; id < 3; ++id)
                            screen[(i0*Sj + j0)*3 + id] = COLOR[pixel*6 + id];
                    else screen[(i0*Sj + j0)*3] = pixel;
                    return;
                }
            }
        }



        # define depth 8
        __kernel void proj_optimized(\
            __global int*   world,    __global int*   worldshape,\
            __global int*   screen,   __global int*   COLOR,     \
            __global int*   position, __global float* direction, \
            __global float* i_dir,    __global float* j_dir)
        {
            // 0 for test, 1 for plot
            int is_RGB = 1;
            // position on screen
            int i0 = get_global_id(0);
            int j0 = get_global_id(1);
            int Si = get_global_size(0);
            int Sj = get_global_size(1);
            int k0 = get_global_id(2);
            // int const depth = get_global_size(2);

            // load parameters
            int Lx = worldshape[0], Ly = worldshape[1], Lz = worldshape[2];

            /***********************************/
            // ray tracer
            /***********************************/

            // get ray_direction
            int   ray[3];
            float ray_dir[3];
            float view = 1.5;
            for (int id = 0; id < 3; ++id)
                ray_dir[id] = direction[id]+ \
                              view*(i0-0.5*Si)/Sj * i_dir[id] + view*(j0-0.5*Sj)/Sj * j_dir[id];

            // query each pixel on the ray
            // T = sqrt(Lx*Lx+Ly*Ly+Lz*Lz), T is 1.71 Lx
            int T = 1.71*Lx;
            int Tstart = T/depth*k0;
            int Tend   = T/depth*(k0+1);
            int pixel  = 0;
            bool is_out = 0;

            /***********************************/
            // shared memory!
            /***********************************/
            __local int ray_cach[depth];
            ray_cach[k0] = 0;
            for (int t = Tstart; t < Tend; ++t)
            {
                for (int id = 0; id < 3; ++id)
                {
                    ray[id] = position[id]+round(t*ray_dir[id]);
                    if (ray[id]<0 || ray[id]>=worldshape[id]) is_out = 1;
                }
                if (is_out) {break;}

                //  pixel = world(ray[0], ray[1], ray[2])
                pixel = world[ray[0]*Ly*Lz + ray[1]*Lz +ray[2]];
                if (pixel) {ray_cach[k0] = pixel; break;}
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // query cache
            if (k0==0){
                for (int t = 0; t < depth; ++t)
                {
                    pixel = ray_cach[t];
                    if (pixel)
                    {
                        for (int id = 0; id < 3; ++id)
                            screen[(i0*Sj + j0)*3 + id] = COLOR[pixel*6 + id];
                        return;
                    }
                }
                for (int id = 0; id < 3; ++id)
                    screen[(i0*Sj + j0)*3 + id] = COLOR[id];
            }
        }


        __kernel void proj_colorful(\
            __global int*   world,    __global int*   worldshape,\
            __global int*   screen,   __global int*   COLOR,     \
            __global int*   position, __global float* direction, \
            __global float* i_dir,    __global float* j_dir)
        {
            // 0 for test, 1 for plot
            int is_RGB = 1;
            // position on screen
            int i0 = get_global_id(0);
            int j0 = get_global_id(1);
            int Si = get_global_size(0);
            int Sj = get_global_size(1);

            // load parameters
            int Lx = worldshape[0], Ly = worldshape[1], Lz = worldshape[2];

            /***********************************/
            // ray tracer
            /***********************************/

            // get ray_direction
            int   ray[3];
            float ray_dir[3];
            float view = 1.5;
            for (int id = 0; id < 3; ++id)
                ray_dir[id] = direction[id]+ \
                              view*(i0-0.5*Si)/Sj * i_dir[id] + view*(j0-0.5*Sj)/Sj * j_dir[id];

            // query each pixel on the ray
            // T = sqrt(Lx*Lx+Ly*Ly+Lz*Lz), T is 1.71 Lx
            int T = 1.71*Lx;
            bool is_out = 0;
            for (int t = 0; t < T; ++t)
            {
                for (int id = 0; id < 3; ++id)
                {
                    ray[id] = position[id]+round(t*ray_dir[id]);
                    if (ray[id]<0 || ray[id]>=worldshape[id]) is_out = 1;
                }
                if (is_out){
                    // set background color to air
                    float r1 = (float)(ray[0]+ray[1])/Lz/2;
                    if (is_RGB)
                        for (int id = 0; id < 3; ++id)
                            screen[(i0*Sj + j0)*3 + id] = (int) (r1*COLOR[id]+(1-r1)*COLOR[id+3]);
                    else screen[(i0*Sj + j0)*3] = 0;
                    return;
                }

                //  pixel = world(ray[0], ray[1], ray[2])
                int pixel = world[ray[0]*Ly*Lz + ray[1]*Lz +ray[2]];
                if (pixel)
                {
                    float r2 = (float)(ray[0]+ray[1])/Lz/2;
                    if (is_RGB)
                        for (int id = 0; id < 3; ++id)
                            screen[(i0*Sj + j0)*3 + id] = \
                                (int) (r2*COLOR[pixel*6 + id] + (1-r2)*COLOR[pixel*6 + id + 3]);
                    else screen[(i0*Sj + j0)*3] = pixel;
                    return;
                }
            }
        }
        """
        self.prg = cl.Program(ctx,kernel_code).build()


    def proj(self, position, direction, flag):
        # given position & direction of UAV, return the projection
        z_dir = np.array([0, 0, 1])
        j_dir = np.cross(direction, z_dir)
        j_dir = j_dir/ np.linalg.norm(j_dir)
        i_dir = np.cross(direction, j_dir)

        i_dir_gpu   = cl.array.to_device(self.queue, i_dir.astype(np.float32))
        j_dir_gpu   = cl.array.to_device(self.queue, j_dir.astype(np.float32))
        pos_gpu     = cl.array.to_device(self.queue, position.astype(np.int32))
        dir_gpu     = cl.array.to_device(self.queue, direction.astype(np.float32))
        
        # set to different mode
        Si, Sj = self.screenshape[0:2]
        if flag == 'naive':
            func        = self.prg.proj_naive
            localshape  = (1, 1)
            globalshape = (Si, Sj)
        if flag == 'colorful':
            func        = self.prg.proj_colorful
            localshape  = (1, 1)
            globalshape = (Si, Sj)
        if flag == 'optimized':
            func        = self.prg.proj_optimized
            # depth       = np.sqrt(self.worldshape[0]).astype(np.int32)
            depth = 8
            localshape  = (1, 1, depth)
            globalshape = (Si, Sj, depth)

        event = func(self.queue, globalshape, localshape,
                    self.world_gpu.data,    self.worldshape_gpu.data,
                    self.screen_gpu.data,   self.COLOR_gpu.data,
                    pos_gpu.data,           dir_gpu.data,
                    i_dir_gpu.data,         j_dir_gpu.data)
        event.wait()
        runtime = 1e-10 * (event.profile.end-event.profile.start)

        return self.screen_gpu.get(), runtime
