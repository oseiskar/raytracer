
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

class Sphere(Tracer):
	
	extra_normal_argument_definitions = ["const float3 center", "const float invR"]
	extra_tracer_argument_definitions = ["const float3 center", "const float R2"]
		
	@property
	def extra_normal_arguments(self):
		return ["(float3)%s" % (self.pos,), 1.0/self.R]
		
	@property
	def extra_tracer_arguments(self):
		return ["(float3)%s" % (self.pos,), self.R**2]
	
	def __init__(self, pos, R):
		self.pos = tuple(pos)
		self.R = R
	
	tracer_code = """
		
		if (origin_self && !inside)
		{
			// convex body
			return;
		}
		
		float3 rel = center - origin;
		float dotp = dot(ray, rel);
		float psq = dot(rel, rel);
		
		float dist, discr, sqrdiscr;
		
		if (dotp <= 0 && !inside)
		{
			// ray travelling away from the center, not starting inside 
			// the sphere => no intersection
			return;
		}
		
		discr = dotp*dotp - psq + R2;
		if(discr < 0) return;
		
		sqrdiscr = native_sqrt(discr);
		dist = dotp - sqrdiscr;
		
		if (inside) dist = dotp + sqrdiscr;
		else dist = dotp - sqrdiscr;
		
		if (dist <= 0) return;
		*p_new_isec_dist = dist;
		"""
	
	normal_code = "*p_normal = (pos - center) * invR;"
	
	@staticmethod
	def get_bounding_volume_code(center, R, minvar, maxvar):
		if R == None:
			code = """
			%s = 0.0f;
			%s = old_isec_dist;
			""" % (minvar, maxvar)
		else:
			code =  """
			{
			// Bounding sphere intersection
			
			const float R2 = %s;
			const float3 center = (float3)%s;
			float3 rel = center - origin;
			float dotp = dot(ray, rel);
			float psq = dot(rel, rel);
			""" % (R**2, tuple(center))
			
			code += """
			bool inside_bnd = psq < R2;
			
			if (dotp <= 0 && !inside_bnd) return;
			
			const float discr = dotp*dotp - psq + R2;
			if(discr < 0) return;
			const float sqrdiscr = native_sqrt(discr);
			
			%s = max(dotp-sqrdiscr,0.0f);
			%s = min(dotp+sqrdiscr,old_isec_dist);
			""" % (minvar,maxvar)
			
			code += """
			if (%s <= %s) return;
			}
			"""  % (maxvar, minvar)
		
		return code

class Cylinder(Tracer):
	"""
	Capped cylinder
	"""
	
	extra_tracer_argument_definitions = [
		"const float3 bottom_center",
		"const float3 axis",
		"const float height",
		"const float R2"]
	extra_normal_argument_definitions = [
		"const float3 bottom_center",
		"const float3 axis",
		"const float height",
		"const float invR"]
		
	@property
	def extra_tracer_arguments(self):
		return [
			"(float3)%s" % (self.bottom_center,),
			"(float3)%s" % (self.axis,),
			self.height,
			self.R**2]
		
	@property
	def extra_normal_arguments(self):
		return [
			"(float3)%s" % (self.bottom_center,),
			"(float3)%s" % (self.axis,),
			self.height,
			1.0/self.R]
	
	def __init__(self, bottom_center, axis, height, R):
		"""
		axis is normalized to a unit vector
		"""
		
		self.bottom_center = tuple(bottom_center)
		self.axis = normalize_tuple(axis)
		self.height = height
		self.R = R
	
	tracer_code = """
		
		if (origin_self && !inside)
		{
			// convex body
			return;
		}
		
		float3 rel = origin - bottom_center;
		float z0 = dot(rel,axis), zslope = dot(ray,axis);
		
		if (!inside && ((z0 < 0 && zslope < 0) || (z0 > height && zslope > 0)))
		{
			// outside, not between the cap planes and travelling
			// away from the planes
			return;
		}
		
		float3 perp = rel - z0*axis;
		float3 ray_perp = ray - zslope*axis;
		
		float dotp = dot(ray_perp,perp);
		
		float perp2 = dot(perp,perp);
		float ray_perp2 = dot(ray_perp,ray_perp);
		
		float discr = dotp*dotp - ray_perp2*(perp2 - R2);
		
		if (discr < 0)
		{
			// ray does not hit the infinite cylinder
			return;
		}
		
		// ray hits the infinite cylinder
		
		float sqrtdiscr = native_sqrt(discr);
		float dist = -dotp - sqrtdiscr;
		
		if (inside) dist += 2*sqrtdiscr;
		dist /= ray_perp2;
		
		float z = z0 + dist*zslope;
		float zplane = 0;
		float zplane_dist;
		*p_subobject = 1;
		
		if (inside || (z0 >= 0 && z0 <= height))
		{
			if (zslope > 0)
			{
				zplane = height;
				*p_subobject = 2;
			}
			
			zplane_dist = (zplane-z0)/zslope;
			if (dist < zplane_dist)
			{
				*p_subobject = 0;
			}
			else
			{
				if (inside)
				{
					dist = zplane_dist;
				}
				else
				{
					return;
				}
			}
		}
		else
		{
			if (z0 > height)
			{
				zplane = height;
				*p_subobject = 2;
			}
			
			zplane_dist = (zplane-z0)/zslope;
			if (z >= 0 && z <= height)
			{
				*p_subobject = 0;
			}
			else
			{
				if (dist <= 0)
				{
					dist += 2*sqrtdiscr/ray_perp2;
					if (dist < zplane_dist) return;
					else
					{
						dist = zplane_dist;
					}
				}
				else
				{
					if ((z0 < 0 && z > height) || (z0 > height && z < 0)) return;
					else
					{
						dist += 2*sqrtdiscr/ray_perp2;
						z = z0 + dist*zslope;
						if ((z0 > height && z < height) || (z0 < 0 && z > 0))
						{
							dist = zplane_dist;
						}
						else return;
					}
				}
			}
		}
		
		*p_new_isec_dist = dist;
		"""
	
	normal_code =  """
		if (subobject == 0)
		{
			float3 rel = pos - bottom_center;
			float3 perp = rel - dot(rel,axis)*axis;
			*p_normal = perp * invR;
		}
		else if (subobject == 1)
		{
			*p_normal = -axis;
		}
		else
		{
			*p_normal = axis;
		}
		"""

class Cone(Tracer):
	
	extra_tracer_argument_definitions = [
		"const float3 tip",
		"const float3 axis",
		"const float height",
		"const float s2"]
		
	extra_normal_argument_definitions = [
		"const float3 tip",
		"const float3 axis",
		"const float slope"]
	
	@property
	def extra_tracer_arguments(self):
		return [
			"(float3)%s" % (self.tip,),
			"(float3)%s" % (self.axis,),
			self.height,
			self.R**2 / float(self.height**2)]
		
	@property
	def extra_normal_arguments(self):
		return [
			"(float3)%s" % (self.tip,),
			"(float3)%s" % (self.axis,),
			self.R / float(self.height)]
	
	def __init__(self, tip, axis, height, R):
		self.tip = tuple(tip)
		self.axis = normalize_tuple(axis)
		self.height = height
		self.R = R
	
	tracer_code = """
		
		if (origin_self && !inside)
		{
			// convex body
			return;
		}
		
		float3 rel = origin - tip;
		float z0 = dot(rel,axis);
		float ray_par_len = dot(ray,axis);
		
		if (!inside)
		{
			if (z0 < 0 && ray_par_len < 0)
			{
				// over the tip, travelling up
				return;
			}
			if (z0 > height && ray_par_len > 0)
			{
				// below bottom, travelling down
				return;
			}
		}
		
		float3 rel_par = z0*axis;
		float3 rel_perp = rel - rel_par;
		float3 ray_par = ray_par_len*axis;
		float3 ray_perp = ray - ray_par;
		float ray_perp_len2 = dot(ray_perp,ray_perp);
		float rel_perp_len2 = dot(rel_perp,rel_perp);
		
		float a = ray_perp_len2 - s2*ray_par_len*ray_par_len;
		float hb = dot(rel_perp,ray_perp) - s2 * ray_par_len*z0;
		float c = rel_perp_len2 - s2 * z0*z0;
		
		float discr = hb*hb - a*c;
		if (discr < 0) return; // no intersection with infinite cylinder
		
		float sqrtdiscr = native_sqrt(discr);
		float dist1, dist2, dist =  (-hb - sqrtdiscr)/a, z;
		
		if (a >= 0)
		{
			dist1 = dist;
			dist2 = dist + 2*sqrtdiscr/a;
		}
		else
		{
			dist2 = dist;
			dist1 = dist + 2*sqrtdiscr/a;
		}
		
		float zplane_dist = (height-z0)/ray_par_len;
		*p_subobject = 0;
		
		if (inside || z0 <= height)
		{
			if (inside)
			{
				if (a <= 0)
				{
					if (ray_par_len > 0)
					{
						dist = zplane_dist;
						*p_subobject = 1;
					}
					else dist = dist1;
				}
				else
				{
					dist = dist2;
					if (ray_par_len > 0 && (dist > zplane_dist))
					{
						dist = zplane_dist;
						*p_subobject = 1;
					}
				}
			}
			else
			{
				// dist = dist
				z = z0 + dist*ray_par_len;
				if (z < 0 || z > height) return; // dist < 0 already excluded
				// else certain hit
			}
		}
		else
		{
			dist = dist1;
			z = z0 + dist*ray_par_len;
			
			if (z >= 0 && z <= height)
			{
				if (a < 0)
				{
					// dist < 0 already excluded
					dist = zplane_dist;
					*p_subobject = 1;
				}
				// else certain hit to the side
			}
			else
			{
				dist = dist2;
				if (dist < 0) return;
				
				z = z0 + dist*ray_par_len;
				
				if (z >= 0 && z <= height)
				{
					dist = zplane_dist;
					*p_subobject = 1;
				}
				else return;
			}
		}
		
		*p_new_isec_dist = dist;
		"""
	
	normal_code =  """
		if (subobject == 1)
		{
			*p_normal = axis;
			return;
		}
		
		float3 rel = pos - tip;
		float z0 = dot(rel,axis);
		float3 rel_perp = rel - z0*axis;
		
		*p_normal = fast_normalize(rel_perp / (slope * z0) - axis * slope); // TODO
		"""

class ConvexIntersection(Tracer):
	
	class Component(Tracer):
		base_tracer_argument_definitions = [ \
			# ray origin
			"const float3 origin",        
			# ray direction
			"const float3 ray",
			# upper bound for isec. distance
			"const float old_isec_dist",  
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
			%s(rel, ray, old_isec_dist, &cur_ibegin, &cur_iend, &cur_subobj, inside, %s);
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
	

class LayerComponent(ConvexIntersection.Component):
	
	extra_tracer_argument_definitions = ['const float3 uax', 'const float h']
	extra_normal_argument_definitions = ['const float3 uax']
	
	def __init__(self, axis):
		self.uax = normalize_tuple(axis)
		self.h = vec_norm(axis)
		
		self.extra_tracer_arguments = ["(float3)%s" % (self.uax,), self.h]
		self.extra_normal_arguments = ["(float3)%s" % (self.uax,)]
	
	n_subobjects = 2
	
	tracer_code = """
		float slope = dot(ray,uax), d1, d2;
		
		if (slope > 0) {
			d1 = dot(-origin, uax);
			d2 = d1 + h;
		}
		else {
			d2 = dot(-origin, uax);
			d1 = d2 + h;
		}
		
		if ( (slope > 0) == inside ) *p_subobject = 1;
		else *p_subobject = 0;
		
		*p_isec_begin = d1 / slope;
		*p_isec_end = d2 / slope;
		"""
	
	normal_code = """
		if (subobject == 1) *p_normal = uax;
		else *p_normal = -uax;
		"""
	
	

class Parallelepiped(ConvexIntersection):
	
	def __init__(self, origin, ax1, ax2, ax3):
		components = [ LayerComponent(ax) for ax in (ax1,ax2,ax3)]
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = ''
	

class HalfSpace(Tracer):
	
	extra_normal_argument_definitions = ["const float3 normal"]
	extra_tracer_argument_definitions = ["const float3 normal", "const float h"]
	
	def __init__(self, normal, h):
		self.normal_vec = normalize_tuple(normal)
		self.h = h
		
		self.extra_tracer_arguments = ["(float3)%s" % (self.normal_vec,), self.h]
		self.extra_normal_arguments = ["(float3)%s" % (self.normal_vec,)]
	
	tracer_code = """
		if (!origin_self)
		{
			float slope = dot(ray,-normal);
			float dist = dot(origin, normal)+h;
			
			dist = dist/slope;
			if (dist > 0) *p_new_isec_dist = dist;
		}
		"""
	
	normal_code = "*p_normal = normal;"

class ImplicitSurface(Tracer):

	# The equation defining the surface must be positive outside the object
	# (multiply eq. by -1 if things do not work)
	
	def __init__(self, eq,
				center=(0,0,0), scale=1.0, bndR=None,
				max_itr=1000, precision=0.001):
		
		self.unique_tracer_id = str(id(self))
		
		self.center = tuple(center)
		self.scale = scale
		
		import sympy
		import sympy.core.numbers
		
		x,y,z = sympy.symbols('x y z')
		
		self.eq = sympy.sympify(eq).subs([\
			(x,((x-self.center[0])/self.scale)),
			(y,((y-self.center[1])/self.scale)),
			(z,((z-self.center[2])/self.scale))])
		
		self.gx = sympy.diff(self.eq,x)
		self.gy = sympy.diff(self.eq,y)
		self.gz = sympy.diff(self.eq,z)
		
		#print eq
		#print gx
		#print gy
		#print gz
		
		# Must replace some expressions to make it OpenCL
		class Printer(sympy.printing.str.StrPrinter):
			def _print_Pow(self, expr):
				base = expr.args[0]
				exponent = expr.args[1]
				f = "pow"
				if exponent.is_integer:
					# pown was really slow on my gpu...
					n = int(exponent)
					if n > 0:
						base_str = "("+str(base)+")"
						return "("+('*'.join([base_str]*n))+")" 
					
				return "%s(%s,%s)" % (f, base, exponent)
		
		# Print as an interval arithmetic macro expression
		class IAPrinter(sympy.printing.str.StrPrinter):
			
			def _print_Pow(self, expr):
				base = expr.args[0]
				exponent = expr.args[1]
				return "ia_pow%d(%s)" % (int(exponent), base)
				
			
			def _print_mul_rec(self,args):
				
				a = args[0]
				b = args[1]
				
				if b.is_number: a,b = b,a
				
				if len(args) > 2:
					bstr = self._print_mul_rec(args[1:])
				else:
					bstr = str(b)
				
				if a.is_number:	
					if b.is_number: return "((%s)*(%s))" % (a,b)
					if a >= 0:
						return "ia_mul_pos_exact(%s,%s)" % (bstr,a)
					else:
						return "ia_mul_neg_exact(%s,%s)" % (bstr,a)
				else:
					return "ia_mul(%s,%s)" % (a,bstr)
			
			def _print_Mul(self, expr):
				return self._print_mul_rec(expr.args)
				
		
		old_ptr = sympy.Basic.__str__
		sympy.Basic.__str__ = lambda self: IAPrinter().doprint(self)
		
		self.tracer_code = "ia_type cur_ival;"
		
		self.tracer_code += Sphere.get_bounding_volume_code(\
			self.center, bndR*self.scale, 'ia_begin(cur_ival)', 'ia_end(cur_ival)');
		
		self.tracer_code += """
		
		int i=0;
		const int MAX_ITER = %d;
		const float TARGET_EPS = %s;
		const float FRACTION = 0.5;
		const float SELF_MAX_BEGIN_STEP = 0.01;
		""" % (max_itr, precision)
		
		self.tracer_code += """
		ia_type x, y, z, f, df;
		
		float step;
		int steps_since_subdiv = 0;
		
		if (origin_self)
		{
			ia_end(cur_ival) = SELF_MAX_BEGIN_STEP;
		}

		for( i=0; i < MAX_ITER; i++ )
		{
			if (ia_begin(cur_ival) >= old_isec_dist) return;
			if (ia_end(cur_ival) > old_isec_dist) ia_end(cur_ival) = old_isec_dist;
			
			x = ia_add_exact(ia_mul_exact(cur_ival, ray.x), origin.x);
			y = ia_add_exact(ia_mul_exact(cur_ival, ray.y), origin.y);
			z = ia_add_exact(ia_mul_exact(cur_ival, ray.z), origin.z);
			
			%s
			
			step = ia_len(cur_ival);
			
			if (ia_contains_zero(f))
			//if (ia_begin(f) < 0)
			{
				if (step < TARGET_EPS || i == MAX_ITER-1)
				{
					step = ia_center(cur_ival);
					*p_new_isec_dist = step;
					return;
				}
				
				//if (origin_self)
				{
					%s
				}
				
				//if ( !origin_self || (ia_begin(df)<0) != inside )
				if ( (ia_begin(df)<0) != inside )
				{
					// Subdivide
					step *= FRACTION;
					ia_end(cur_ival) = ia_begin(cur_ival) + step;
					steps_since_subdiv = 0;
					continue;
				}
			}
			steps_since_subdiv++;
			
			// Step forward
			if (steps_since_subdiv > 1) step /= FRACTION;
			cur_ival = ia_new(ia_end(cur_ival),ia_end(cur_ival)+step);
		}
		""" % (self.compute_f_code(), self.compute_df_code());
		
		#print eq
		#print gx
		#print gy
		#print gz
		
		sympy.Basic.__str__ = lambda self: Printer().doprint(self)
		
		x.name = 'pos.x'
		y.name = 'pos.y'
		z.name = 'pos.z'
		self.normal_code = self.compute_normal_code()
		
		sympy.Basic.__str__ = old_ptr
		
		#print self.tracer_code
		
	def compute_f_code(self):
		return "f = %s;" % self.eq
	
	def compute_df_code(self):
		return """
			df = ia_add(
				ia_add(
					ia_mul_exact(%s,ray.x),
					ia_mul_exact(%s,ray.y)),
				ia_mul_exact(%s,ray.z));
		""" % (self.gx,self.gy,self.gz)
	
	def compute_normal_code(self):
		return """
		*p_normal = fast_normalize((float3)(%s, %s, %s));
		""" % (self.gx,self.gy,self.gz)

class QuaternionJuliaSet2(Tracer):
	
	def __init__(self, c, julia_itr, **kwargs):
		self.c = c
		self.julia_itr = julia_itr
		self.bndR = 1.5
		self.center = (0,0,0)
		self.max_itr = 100
		self.precision = 0.001
		self.max_step = 0.1
		
		# TODO: tuple(center)
		
		for (k,v) in kwargs.items(): setattr(self, k, v)
		
		self.tracer_code = """
		if (origin_self) return; // ------------- remember to remove this
		
		float trace_begin, trace_end;
		"""
		
		self.tracer_code += Sphere.get_bounding_volume_code(\
			self.center, self.bndR, 'trace_begin', 'trace_end');
		
		self.tracer_code += """
		
		int i=0;
		const int MAX_ITER = %d;
		const float TARGET_EPS = %s;
		const float MAX_STEP = %s;
		const float ESCAPE_RADIUS = 2.0f;
		""" % (self.max_itr, self.precision, self.max_step)
		
		self.tracer_code += """
		float step, dist;
		const float4 c = (float4)%s;
		dist = trace_begin;
		float3 pos = trace_begin * ray + origin - (float3)%s;
		
		""" % (self.c,self.center)
		
		self.tracer_code += """
		for( i=0; i < MAX_ITER; i++ )
		{
			float4 q = (float4)(pos,0), q1, qd = (float4)(1,0,0,0);
			
			float gr = ray.x, gi = ray.y, gj = ray.z, gk = 0,
				  gr1, gi1, gj1, gk1;
			
			for (int iii=0; iii<%d; ++iii)
			{
				if (origin_self)
				{
					// TODO: use quaternion derivative, if possible
				
					// Derivative chain rule
					gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
					gi1 = 2*(gr*q.y + q.x*gi);
					gj1 = 2*(gr*q.z + q.x*gj);
					gk1 = 2*(gr*q.w + q.x*gk);
					
					gr = gr1;
					gi = gi1;
					gj = gj1;
					gk = gk1;
				}
				
				qd = 2*quaternion_mult(q,qd);
				q = quaternion_square(q) + c;
				
			}
			
			float ql2 = dot(q,q);
			float ds = 2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk);
			
			// The magic distance estimate formula, see the 1989 article:
			// Hart, Sandin, Kauffman, "Ray Tracing Deterministic 3-D Fractals"
			float l = length(q);
			step = 0.5 * l * log(l) / length(qd);
			step = min(step, MAX_STEP);
			
			if (step < TARGET_EPS)
			{
				if (!origin_self || ds < 0)
				{ 
					*p_new_isec_dist = dist + step;
					return;
				}
				else
				{
					step = TARGET_EPS;
				}
			}
			/*if (ql2 < ESCAPE_RADIUS*ESCAPE_RADIUS)
			{
				*p_new_isec_dist = dist;
				return;
			}*/
			
			dist += step;
			
			if (dist > trace_end) return;
			pos += step * ray;
		}
		""" % self.julia_itr
		
		self.normal_code = """
		float4 q = (float4)(pos - (float3)%s,0);
		const float4 c = (float4)%s, q1;
		""" % (self.center,self.c)
		
		self.normal_code += """
		float3
			gr = (float3)(1,0,0),
			gi = (float3)(0,1,0),
			gj = (float3)(0,0,1),
			gk = (float3)(0,0,0),
			gr1, gi1, gj1, gk1;
		
		float qr1;
		
		for (int iii=0; iii<%d; ++iii)
		{
			// Derivative chain rule
			gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
			gi1 = 2*(gr*q.y + q.x*gi);
			gj1 = 2*(gr*q.z + q.x*gj);
			gk1 = 2*(gr*q.w + q.x*gk);
			
			gr = gr1;
			gi = gi1;
			gj = gj1;
			gk = gk1;
			
			// Quaternion operation z -> z^2 + c
			q = quaternion_square(q) + c;
		}
		float3 grad = fast_normalize(2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk));
		
		if (all(isfinite(grad))) *p_normal = grad;
		else *p_normal = (float3)(1,0,0);
		""" % self.julia_itr
	


class QuaternionJuliaSet(ImplicitSurface):
	
	def __init__(self, c, julia_itr, *args, **argd):
		self.c = c
		self.julia_itr = julia_itr
		ImplicitSurface.__init__(self, "x^2 + y^2 + z^2 - 1", *args, **argd)
		
		self.tracer_code = """
		if (origin_self) return;
		""" + self.tracer_code
	
	def compute_f_code(self):
		
		return """
		ia_type qr = x - %s;
		ia_type qi = y - %s;
		ia_type qj = z - %s;
		qr /= %s;
		qi /= %s;
		qj /= %s;
		ia_type qk = ia_new(0,0);
		
		const float cr = %s, ci = %s, cj = %s, ck = %s;
		ia_type qr1;
		
		for (int iii=0; iii<%d; ++iii)
		{
			// Quaternion operation z -> z^2 + c
			// lazy... "should" use ia_add
			qr1 = ia_sub(ia_pow2(qr), ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk))+cr;
			qi = 2 * ia_mul(qr,qi) + ci;
			qj = 2 * ia_mul(qr,qj) + cj;
			qk = 2 * ia_mul(qr,qk) + ck;
			qr = qr1;
		}
		
		f = ia_pow2(qr)+ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk) - 4.0;
		""" % (self.center+tuple([self.scale]*3)+self.c+(self.julia_itr,))
		
		return "f = %s;" % self.eq
	
	def compute_df_code(self): # TODO
		return """
			df = 1;
		"""
	
	def compute_normal_code(self):
		return """
		
		float qr = pos.x - %s;
		float qi = pos.y - %s;
		float qj = pos.z - %s;
		float qk = 0;
		qr /= %s;
		qi /= %s;
		qj /= %s;
		
		float3
			gr = (float3)(1,0,0),
			gi = (float3)(0,1,0),
			gj = (float3)(0,0,1),
			gk = (float3)(0,0,0),
			gr1, gi1, gj1, gk1;
		
		const float cr = %s, ci = %s, cj = %s, ck = %s;
		float qr1;
		
		for (int iii=0; iii<%d; ++iii)
		{
			// Derivative chain rule...
			
			gr1 = 2*(qr*gr - qi*gi - qj*gj - qk*gk);
			gi1 = 2*(gr*qi + qr*gi);
			gj1 = 2*(gr*qj + qr*gj);
			gk1 = 2*(gr*qk + qr*gk);
			
			// Quaternion operation z -> z^2 + c
			qr1 = qr*qr - qi*qi - qj*qj - qk*qk + cr;
			qi = 2 * qr*qi + ci;
			qj = 2 * qr*qj + cj;
			qk = 2 * qr*qk + ck;
			qr = qr1;
			
			gr = gr1;
			gi = gi1;
			gj = gj1;
			gk = gk1;
		}
		
		//const float3 grad = gr*qr;
		float3 grad = fast_normalize(2*(qr*gr + qi*gi + qj*gj + qk*gk));
		
		if (all(isfinite(grad))) *p_normal = grad;
		else *p_normal = (float3)(1,0,0);
		
		//const float l = length(grad);
		//if (l==0) *p_normal = (1,0,0);
		//else *p_normal = grad/l;
		//*p_normal = (float3)(1,0,0);
		//*p_normal = fast_normalize((float3)(1,gr.x*gr.x*0.001,0));
		
		//*p_normal = fast_normalize((float3)(dx, dy, dz));
		""" % (self.center+tuple([self.scale]*3)+self.c+(self.julia_itr,))

