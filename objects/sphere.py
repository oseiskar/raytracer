from tracer import Tracer
from utils import normalize_tuple, vec_norm
import numpy
import math

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
    
	def surface_area(self):
		return 4.0 * math.pi * self.R**2

	def random_surface_point_and_normal(self):
		p = numpy.array(self.pos)
		rand = numpy.random.normal(0,1,p.shape)
		rand = rand / numpy.linalg.norm(rand)
		return (p + rand * self.R, rand)
	
	def center_and_min_sampling_distance(self):
		return (numpy.array(self.pos), self.R * 2.0)
	
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
