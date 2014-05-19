
from utils import normalize_tuple, vec_norm
import numpy
import sys

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

class ConvexIntersection(Tracer):
	"""Intersection of convex objects represented by Components"""
	
	class Component(Tracer):
		base_tracer_argument_definitions = [ \
			# ray origin
			"const float3 origin",        
			# ray direction
			"const float3 ray",
			# [out] computed isec. interval begin distance
			"__private float *p_isec_begin",  
			# [out] computed isec. interval end distance
			"__private float *p_isec_end",  
			# [out] (optional) subobject number (e.g., which face of a cube),
			# passed to the normal computation function
			"__private uint *p_subobject", 
			# is the ray travelling inside the object
			"bool inside"]
	
	def make_functions( self ):
		funcs = Tracer.make_functions( self )
		for component in self.components:
			subfuncs = component.make_functions()
			funcs = dict( funcs.items() + subfuncs.items() )
		return funcs
	
	def __init__(self, origin, components):
		self.origin = origin
		self.components = components
		
		self.extra_tracer_argument_definitions = ['const float3 base']
		self.extra_tracer_arguments = ["(float3)%s" % (self.origin,)]
		
		self.extra_normal_argument_definitions = ['const float3 base']
		self.extra_normal_arguments = ["(float3)%s" % (self.origin,)]
		
		self.unique_tracer_id = str(id(self))
		
		self.tracer_args_per_component = []
		self.normal_args_per_component = []
		self.subobjects_per_component = []
		
		n_tracer_extra_args = 0
		n_normal_extra_args = 0
		for c in self.components:
			tdefs = c.extra_tracer_argument_definitions
			ndefs = c.extra_normal_argument_definitions
			
			for td in tdefs:
				parts = td.split(' ')
				parts[-1] = 'arg_%d' % n_tracer_extra_args
				self.extra_tracer_argument_definitions.append( ' '.join( parts ) )
				n_tracer_extra_args += 1
			
			for nd in ndefs:
				parts = nd.split(' ')
				parts[-1] = 'arg_%d' % n_normal_extra_args
				self.extra_normal_argument_definitions.append( ' '.join( parts ) )
				n_normal_extra_args += 1
			
			self.extra_tracer_arguments += c.extra_tracer_arguments
			self.extra_normal_arguments += c.extra_normal_arguments
	
	@property
	def tracer_code(self):
		s =	"""
		if (origin_self && !inside) return;
		
		float3 rel = origin - base;
		float ibegin  = 0.0, iend = old_isec_dist;
		float cur_ibegin, cur_iend;
		uint subobj, cur_subobj;
		"""
		
		arg_offset = 0
		subobj_offset = 0
		for c_idx in range(len(self.components)):
			c = self.components[c_idx]
			fname = c.tracer_function_name
			n_arg = len( c.extra_tracer_argument_definitions )
			extra_args = ', '.join(['arg_%d' % (arg_offset+i) for i in range(n_arg)])
			s += """
			cur_subobj = 0;
			cur_ibegin = ibegin;
			cur_iend = iend;
			%s(rel, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside, %s);
			""" % (fname, extra_args)
			
			s += """
			if (cur_ibegin > ibegin) {
				ibegin = cur_ibegin;
				if (!inside) subobj = cur_subobj + %d;
			}
			if (cur_iend < iend) {
				iend = cur_iend;
				if (inside) subobj = cur_subobj + %d;
			}
			if (ibegin > iend || ibegin > old_isec_dist) return;
			""" % (subobj_offset, subobj_offset)
			
			arg_offset += n_arg
			subobj_offset += c.n_subobjects
		
		s += """
		if (inside) *p_new_isec_dist = iend;
		else *p_new_isec_dist = ibegin;
		*p_subobject = subobj;
		"""
		
		return s
	
	@property
	def normal_code(self):
		s = """
		const float3 p = pos - base;
		"""
		
		arg_offset = 0
		subobj_offset = 0
		for c in self.components:
			n_arg = len( c.extra_normal_argument_definitions )
			extra_args = ', '.join(['arg_%d' % (arg_offset + i) for i in range(n_arg)])
			
			if subobj_offset > 0:
				s += """
				if (subobject >= %d && subobject < %d)""" \
					% (subobj_offset, subobj_offset+c.n_subobjects)
			else:
				# avoiding compiler warning about 'pointless comparison
				# of unsigned integer with zero'
				s += """
				if (subobject < %d)""" % (subobj_offset+c.n_subobjects)
			
			s += """ %s(p, subobject - %d, p_normal, %s);
			""" % (c.normal_function_name, subobj_offset, extra_args)
			
			arg_offset += n_arg
			subobj_offset += c.n_subobjects
		
		return s
