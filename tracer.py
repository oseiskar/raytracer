
from utils import normalize_tuple, vec_norm
import numpy

class Tracer:
	"""
	A Tracer instance represents the shape of a three-dimensional body.
	It is responsible for generating the OpenCL code that can compute the
	intersection of a ray and this object (given helpful extra information
	that is accumulated during the tracing process) and an exterior normal
	at that intersection.
	"""

	# OpenCL argument definitions of the tracer (intersection) functions
	#
	# The tracer functions are supposed compute the distance to nearest valid
	# intersection of the given ray and the object that the tracer function
	# represents
	base_tracer_argument_definitions = [ \
		# ray origin
		"const float3 origin",        
		# ray direction
		"const float3 ray",           
		# previous normal (for computing self-intersections)
		"const float3 last_normal",   
		# upper bound for isec. distance
		"const float old_isec_dist",  
		# [out] computed isec. distance
		"__private float *p_new_isec_dist",  
		# [out] (optional) subobject number (e.g., which face of a cube),
		# passed to the normal computation function
		"__private uint *p_subobject", 
		# is the ray travelling inside the object
		"bool inside",
		# self-intersection?
		"bool origin_self"]
	
	# OpenCL argument definitions of the normal computation functions
	#
	# The normal functions are supposed to compute the exterior normal
	# at the given point
	#
	base_normal_argument_definitions = [ \
		# a point on the surface of the object
		"const float3 pos",
		# subobject number (computed by the tracer)
		"const uint subobject",
		# [out] the computed normal
		"__global float3 *p_normal" ]
	
	# overriden by subclasses: OpenCL definitions of possible extra arguments
	# to the tracer functions (e.g., the radius of a sphere) 
	extra_tracer_argument_definitions = []
	# OpenCL definitions of extra arguments to the normal functions
	extra_normal_argument_definitions = []
	
	# values of the extra tracer arguments (different for each instance/object)
	extra_tracer_arguments = []
	# values of the normal arguments
	extra_normal_arguments = []
	# setting this to, e.g., id(obj), causes a tracer function to be
	# generated for each instance, instead of one per Tracer (sub)class
	unique_tracer_id = ""
	
	def _function_name_prefix(self):
		return self.__class__.__name__+self.unique_tracer_id
	
	@property
	def tracer_function_name(self):
		return self._function_name_prefix() + '_tracer'
	
	@property
	def normal_function_name(self):
		return self._function_name_prefix() + '_normal'
	
	def make_tracer_function(self):
		tracer_arguments = ",".join(self.base_tracer_argument_definitions \
			+ self.extra_tracer_argument_definitions)
		tracer_function = "void %s(%s) {" \
			% (self.tracer_function_name, tracer_arguments)
		tracer_function += "\n" + self.tracer_code + "\n}\n\n"
		return tracer_function
	
	def make_normal_function(self):
		normal_arguments = ",".join(self.base_normal_argument_definitions \
			+ self.extra_normal_argument_definitions)
		normal_function = "void %s(%s) {" \
			% (self.normal_function_name, normal_arguments)
		normal_function += "\n" + self.normal_code + "\n}\n\n"
		return normal_function
	
	def make_functions(self):
		"""
		Make necessary OpenCL functions for tracing objects of this class
		Returns a dictionary OpenCL function name -> function contents
		"""
		
		return { \
			self.tracer_function_name : self.make_tracer_function(),
			self.normal_function_name : self.make_normal_function()
		}
	
	def make_tracer_call(self, base_params):
		"""
		Make a call that computes the intersection of given ray and an object
		represented by this tracer instance (returns a string of OpenCL code)
		"""
		return "%s(%s);" % (self.tracer_function_name, \
			",".join(base_params+[str(x) for x in self.extra_tracer_arguments]))
	
	def make_normal_call(self, base_params):
		"""
		Make a call that computes an exterior normal in the given intersection
		point (returns a string of OpenCL code)
		"""
		return "%s(%s);" % (self.normal_function_name, \
			",".join(base_params+[str(x) for x in self.extra_normal_arguments]))


