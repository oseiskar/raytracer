
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

class Accelerator:
	"""opencl initialization and some other stuff"""
	
	# OpenCL-related stuff is not well contained here (at the moment)
	# which was the original idea
	
	def __init__(self,interactive=False):
		self.ctx = cl.create_some_context(interactive)
		prop = cl.command_queue_properties.PROFILING_ENABLE
		self.queue = cl.CommandQueue(self.ctx, properties=prop)
	
	def new_array( self, shape, datatype=np.float32, zeros=False ):
		if zeros: ctor = cl_array.zeros
		else: ctor = cl_array.empty
		return ctor(self.queue, shape, dtype=datatype)
		
	def new_vec3_array( self, shape ):
		shape = shape + (4,)
		return cl_array.zeros(self.queue, shape, dtype=np.float32)
	
	def make_vec3_array_xyz( self, x,y,z ):
		
		assert( x.shape == y.shape and y.shape == z.shape )
		
		shape = x.shape + (4,)
		cpuarray = np.empty( shape, dtype=np.float32 )
		cpuarray[...,0] = x
		cpuarray[...,1] = y
		cpuarray[...,2] = z
		cpuarray[...,3] = np.zeros_like(x)
		return cl_array.to_device(self.queue, cpuarray)
		
	def make_vec3_array( self, a ):
		assert( a.shape[-1]==3 )
		
		# TODO: does not work. why?
		"""
		w = np.zeros_like(a[...,0])
		w.shape += (1,)
		cpuarray = np.concatenate( (a, w), axis=len(a.shape)-1 ).astype(np.float32)
		return cl_array.to_device(self.ctx, self.queue, cpuarray)
		"""
		
		return self.make_vec3_array_xyz( a[...,0], a[...,1], a[...,2] )
	
	
	def empty_like( self, a ):
		return cl_array.empty_like(a).with_queue(self.queue)

	def zeros_like( self, a ):
		return cl_array.zeros_like(a).with_queue(self.queue)
