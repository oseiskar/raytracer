
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools
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
        self._scan_kernel = None
        
        devices = self.ctx.get_info(cl.context_info.DEVICES)
        assert(len(devices) == 1)
        self.device = devices[0]
        
        self.profiling_info = {}
        
        try:
            self.warp_size = self.device.get_info(cl.device_info.WARP_SIZE_NV)
            print 'device NV warp size', self.warp_size
        except:
            self.warp_size = 1
    
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
        return self._get_kernel_work_group_info(kernel_name, \
            cl.kernel_work_group_info.WORK_GROUP_SIZE)
    
    def get_preferred_local_work_group_size_multiple(self, kernel_name):
        return self._get_kernel_work_group_info(kernel_name, \
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
        
    def _get_kernel_work_group_info(self, kernel_name, param):
        kernel = getattr(self.prog, kernel_name)
        return kernel.get_work_group_info(param, self.device)
    
    def call(self, kernel_name, ndrange_size, buffer_args, \
            value_args=tuple([]), work_group_size=None):
        
        t1 = time.time()
        kernel = getattr(self.prog, kernel_name)
        
        if isinstance(ndrange_size, int):
            ndrange_size = (ndrange_size,)
        else:
            assert(len(ndrange_size) == 2)
        
        arg = []
        for x in buffer_args:
            if isinstance(x, cl_array.Array):
                x = x.data
            arg.append(x)
        arg = tuple(arg) + value_args
        event = kernel(self.queue, ndrange_size, work_group_size, *arg)
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
    
    def new_local_buffer( self, size ):
        return cl.LocalMemory(size * 4) # TODO: assuming 4-byte types
        
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
    
    class MemPool:
        
        def __init__(self, cl_context):
            self.ctx = cl_context
            self.buffers = []
            self.buffer_idx = 0
        
        def __call__(self, n):
            #print "called buffer", n
            
            if len(self.buffers) <= self.buffer_idx:
                self.buffers.append( cl.Buffer(self.ctx, \
                    cl.mem_flags.READ_WRITE, n ) )
            
            buf = self.buffers[self.buffer_idx]
            size = buf.get_info(cl.mem_info.SIZE)
            
            if size < n:
                raise RuntimeError('size mismatch')
                
            #print 'returning buf', self.buffer_idx, 'of size', size
                
            self.buffer_idx += 1
            return buf
        
        def free(self):
            self.buffer_idx = 0

    def find_non_negative(self, in_array, out_array, n):
        
        from pyopencl.scan import GenericScanKernel
        
        if self._scan_kernel is None:
            # like pyopencl's copy_if kernel
            self._scan_kernel = GenericScanKernel(
                self.ctx, np.int32,
                arguments="__global int *ary, __global int *out, __global int *count",
                input_expr="ary[i] < 0 ? 0 : 1",
                scan_expr="a+b", neutral="0",
                output_statement="""
                    if (prev_item != item) out[item-1] = ary[i];
                    if (i+1 == N) *count = item;
                    """)
            
            self._mem_pool = Accelerator.MemPool(self.ctx)
            
            self._out_int = self.new_array( (1,), np.int32 )
        
        ev = self._scan_kernel(in_array, out_array, self._out_int, \
            size=n, queue=self.queue, allocator=self._mem_pool)
        ev.wait()
        
        self._mem_pool.free()
        
        return int(self._out_int.get()[0])
    
    def device_memcpy(self, dest, src, n=None):
        kwargs = {}
        if n is not None:
            kwargs['byte_count'] = n * 4 # assuming 32-bit values
        ev = cl.enqueue_copy(self.queue, dest.data, src.data, **kwargs)
        ev.wait()
