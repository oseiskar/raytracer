
class Tracer:
	TRACE_KERNEL_ARGUMENTS = """
	__global const float3 *p_origin,
	__global const float3 *p_ray,
	__global const float3 *p_last_normal,
	__global const float *p_old_isec_dist,
	__global float *p_new_isec_dist,
	__global const uint *p_inside,
	__global const uint *p_origin_self""" # TODO: bool
	
	NORMAL_KERNEL_ARGUMENTS = """
	__global const float3 *p_pos,
	__global float3 *p_normal
	"""
	
	TRACE_KERNEL_HEADER = """
	
	const int gid = get_global_id(0);
	const float3 origin = p_origin[gid];
	const float3 ray = p_ray[gid];
	const float3 last_normal = p_last_normal[gid];
	const float old_isec_dist = p_old_isec_dist[gid];
	const bool inside = p_inside[gid] != 0;
	const bool origin_self = p_origin_self[gid] != 0;
	p_new_isec_dist += gid;
	"""
	
	NORMAL_KERNEL_HEADER = """
	
	const int gid = get_global_id(0);
	const float3 pos = p_pos[gid];
	p_normal += gid;
	
	"""
	
	TRACE_STATIC_ARGUMENTS = """
	const float3 origin,
	const float3 ray,
	const float3 last_normal,
	const float old_isec_dist,
	__private float *p_new_isec_dist,
	bool inside,
	bool origin_self"""
	
	NORMAL_STATIC_ARGUMENTS = """
	const float3 pos,
	__global float3 *p_normal
	"""
	
	def make_kernel(self, kernel_function=True):
		
		if kernel_function:
			kernel_keyword = '__kernel'
			trace_arguments = Tracer.TRACE_KERNEL_ARGUMENTS
			normal_arguments = Tracer.NORMAL_KERNEL_ARGUMENTS
			trace_header = Tracer.TRACE_KERNEL_HEADER
			normal_header = Tracer.NORMAL_KERNEL_HEADER
		else:
			kernel_keyword = ''
			trace_arguments = Tracer.TRACE_STATIC_ARGUMENTS
			normal_arguments = Tracer.NORMAL_STATIC_ARGUMENTS
			trace_header = ''
			normal_header = ''
		
		kernel_id = self.__class__.__name__+str(id(self))
		self.tracer_kernel_name = kernel_id+"_tracer"
		self.normal_kernel_name = kernel_id+"_normal"
		
		self.kernel_code = kernel_keyword + " void %s(%s) {" \
			% (self.tracer_kernel_name, trace_arguments)
		self.kernel_code += trace_header
		self.kernel_code += "\n" + self.tracer_code + "\n}\n\n"
		
		self.kernel_code += kernel_keyword + " void %s(%s) {" \
			% (self.normal_kernel_name, normal_arguments)
		self.kernel_code += normal_header
		self.kernel_code += "\n" + self.normal_code + "\n}\n\n"
		
		return self.kernel_code
		

class Sphere(Tracer):
	
	def _params(self): return """
	const float3 center = (float3)%s;
	const float R2 = %s;
	const float invR = %s;
	""" % (self.pos, self.R**2, 1.0/self.R)
	
	@property
	def tracer_code(self): return self._params() + \
	"""
	float3 rel = center - origin;
	float dotp = dot(ray, rel);
	float psq = dot(rel, rel);
	
	float dist, discr, sqrdiscr;
	
	if (origin_self && !inside)
	{
		return;
		// no intersection
	}
	//if (inside) { *p_new_isec_dist = 0.1; return; }
	
	if (dotp <= 0 && !inside)
	{
		// no intersection
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
	
	@property
	def normal_code(self): return self._params() + """
	*p_normal = (pos - center) * invR;
	"""
	
	def __init__(self, pos, R):
		self.pos = pos
		self.R = R

class HalfSpace(Tracer):
		
	def _params(self): return """
	const float3 normal = (float3)%s;
	const float h = %s;
	""" % (self.normal_vec, self.h)
	
	@property
	def tracer_code(self): return self._params() + """
	
	if (!origin_self)
	{
		float slope = dot(ray,-normal);
		float dist = dot(origin, normal)+h;
		
		dist = dist/slope;
		if (dist > 0) *p_new_isec_dist = dist;
	}
	"""
	
	@property
	def normal_code(self): return self._params() + """
	*p_normal = normal;
	"""
	
	def __init__(self, normal, h):
		self.normal_vec = normal
		self.h = h

