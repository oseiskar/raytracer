
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

# Enables a work-around for the PyOpenCL issue #56 in versions 2013 and 2014
FINISH_CL_ARRAYS = cl.VERSION[0] in (2013, 2014)

class Accelerator:
    """
    PyOpenCL initialization, as well as, management of buffers,
    kernel calls and command queues are encapsulated here
    """
    
    def __init__(self, interactive=False):
        self.ctx = cl.create_some_context(interactive)
        prop = cl.command_queue_properties.PROFILING_ENABLE
        self.queue = cl.CommandQueue(self.ctx, properties=prop)
        self._cl_arrays = []
        
        devices = self.ctx.get_info(cl.context_info.DEVICES)
        assert(len(devices) == 1)
        self.device = devices[0]
        
        self.profiling_info = {}
    
    def finish(self):
        """Call finish on relevant CL queues and arrays"""
        self.queue.finish()
        if FINISH_CL_ARRAYS:
            for a in self._cl_arrays:
                a.finish()
    
    def build_program(self, prog_code, options=[]):
        prog_code = prog_code.encode('ascii')
        
        with open('last_code.cl', 'w') as f:
            f.write(prog_code)
            
        self.prog = cl.Program(self.ctx, prog_code).build(options)
    
    def get_max_work_group_size(self, kernel_name):
        return self._get_kernel_work_group_info(\
            cl.kernel_work_group_info.WORK_GROUP_SIZE)
    
    def get_preferred_local_work_group_size_multiple(self, kernel_name):
        return self._get_kernel_work_group_info(\
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
        
    def _get_kernel_work_group_info(self, kernel_name, param):
        kernel = getattr(self.prog, kernel_name)
        return kernel.get_work_group_info(param, self.device)
    
    def call(self, kernel_name, ndrange_size, buffer_args, value_args=tuple([])):
        
        t1 = time.time()
        kernel = getattr(self.prog, kernel_name)
        
        if isinstance(ndrange_size, int):
            ndrange_size = (ndrange_size,)
        
        arg = []
        for x in buffer_args:
            if x is not None:
                x = x.data
            arg.append(x)
        arg = tuple(arg) + value_args
        event = kernel(self.queue, ndrange_size, None, *arg)
        event.wait()
        
        t = (event.profile.end - event.profile.start)
        if kernel_name not in self.profiling_info:
            self.profiling_info[kernel_name] = {'n':0, 't':0, 'ta':0}
            
        self.profiling_info[kernel_name]['t'] += t
        self.profiling_info[kernel_name]['n'] += 1
        self.profiling_info[kernel_name]['ta'] += time.time() - t1
    
    def enqueue_copy( self, dest, src ):
        cl.enqueue_copy(self.queue, dest, src)
    
    def to_device( self, cpuarray ):
        arr = cl_array.to_device(self.queue, cpuarray)
        self._cl_arrays.append(arr)
        return arr
    
    def new_array( self, shape, datatype=np.float32, zeros=False ):
        if zeros:
            ctor = cl_array.zeros
        else:
            ctor = cl_array.empty
        
        arr = ctor(self.queue, shape, dtype=datatype)
        self._cl_arrays.append(arr)
        return arr
    
    def new_const_buffer( self, buf, dtype=np.float32 ):
        mf = cl.mem_flags
        return cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=buf.astype(dtype))
        
    def new_vec3_array( self, shape ):
        shape = shape + (4, )
        arr = cl_array.zeros(self.queue, shape, dtype=np.float32)
        self._cl_arrays.append(arr)
        return arr
    
    def _make_host_vec3_array_xyz( self, x, y, z ):
        assert( x.shape == y.shape and y.shape == z.shape )
        
        shape = x.shape + (4,)
        cpuarray = np.empty( shape, dtype=np.float32 )
        cpuarray[..., 0] = x
        cpuarray[..., 1] = y
        cpuarray[..., 2] = z
        cpuarray[..., 3] = np.zeros_like(x)
        return cpuarray
    
    def make_vec3_array_xyz( self, *args ):
        return self.to_device(self._make_host_vec3_array_xyz(*args))
        
    def make_vec3_array( self, a ):
        assert( a.shape[-1]==3 )
        return self.make_vec3_array_xyz( a[..., 0], a[..., 1], a[..., 2] )
    
    def make_const_vec3_buffer( self, a ):
        assert( a.shape[-1]==3 )
        cpuarray = self._make_host_vec3_array_xyz( a[..., 0], a[..., 1], a[..., 2] )
        return self.new_const_buffer(cpuarray)
    
    def empty_like( self, a ):
        arr = cl_array.empty_like(a)
        self._cl_arrays.append(arr)
        return arr

    def zeros_like( self, a ):
        arr = cl_array.zeros_like(a)
        self._cl_arrays.append(arr)
        return arr
    
    def output_profiling_info(self):
        total = 0
        tatotal = 0
        for (k, v) in self.profiling_info.items():
            t = v['t']*1e-9
            ta = v['ta']
            n = v['n']
            fmt = '%.2g'
            print ('%d\t'+('\t'.join([fmt]*4))+'\t'+k) % (n, t, ta, t/n, ta/n)
            total += t
            tatotal += ta
        print '----', total, 'or', tatotal, 'seconds total'
