
from tracer import Tracer
from utils import normalize_tuple, vec_norm
import numpy

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
		
		def __init__( self, pos = (0,0,0) ):
			self.pos = tuple(pos)
	
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
		self.extra_tracer_arguments = ["(float3)%s" % (tuple(self.origin),)]
		
		self.extra_normal_argument_definitions = ['const float3 base']
		self.extra_normal_arguments = ["(float3)%s" % (tuple(self.origin),)]
		
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
			%s(rel - (float3)%s, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside, %s);
			""" % (fname, c.pos, extra_args)
			
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
			
			s += """ %s(p - (float3)%s, subobject - %d, p_normal, %s);
			""" % (c.normal_function_name, c.pos, subobj_offset, extra_args)
			
			arg_offset += n_arg
			subobj_offset += c.n_subobjects
		
		return s
