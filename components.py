
# Tracer objects: components for objects defined as intersections
# of convex objects (e.g., capped cylinder)

from tracer import *
from utils import normalize_tuple, vec_norm
import numpy
import sys

class HalfSpaceComponent(ConvexIntersection.Component):
	"""Half-space"""
	
	extra_tracer_argument_definitions = ["const float3 normal", "const float h"]
	extra_normal_argument_definitions = ["const float3 normal"]
	
	def __init__(self, normal, h):
		ConvexIntersection.Component.__init__(self)
		self.normal_vec = normalize_tuple(normal)
		self.h = h
		
		self.extra_tracer_arguments = ["(float3)%s" % (self.normal_vec,), self.h]
		self.extra_normal_arguments = ["(float3)%s" % (self.normal_vec,)]
	
	n_subobjects = 1
	
	tracer_code = """
		float slope = dot(ray,normal);
		float dist = dot(-origin,normal)+h;
		
		dist = dist/slope;
		if (slope < 0) *p_isec_begin = dist;
		else *p_isec_end = dist;
		*p_subobject = 0;
		"""
	
	normal_code = "*p_normal = normal;"

class LayerComponent(ConvexIntersection.Component):
	"""Infinite layer with finite thickness"""
	
	extra_tracer_argument_definitions = ['const float3 uax', 'const float h']
	extra_normal_argument_definitions = ['const float3 uax']
	
	def __init__(self, axis, h = None):
		ConvexIntersection.Component.__init__(self)
		self.uax = normalize_tuple(axis)
		if h == None: self.h = vec_norm(axis)
		else: self.h = h
		
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

class SphereComponent(ConvexIntersection.Component):
	"""Sphere"""
	
	extra_normal_argument_definitions = ["const float invR"]
	extra_tracer_argument_definitions = ["const float R2"]
	
	n_subobjects = 1
	
	def __init__(self, pos, R):
		ConvexIntersection.Component.__init__(self,pos)
		self.R = R
		self.extra_normal_arguments = [1.0/self.R]
		self.extra_tracer_arguments =  [self.R**2]
	
	tracer_code = """
		
		float dotp = -dot(ray, origin);
		float psq = dot(origin, origin);
		
		float discr, sqrdiscr;
		
		discr = dotp*dotp - psq + R2;
		
		if(discr < 0) {
			// ray does not hit the sphere
			*p_isec_begin = 1;
			*p_isec_end = 0;
			return;
		}
		
		sqrdiscr = native_sqrt(discr);
		
		*p_isec_begin = dotp - sqrdiscr;
		*p_isec_end = dotp + sqrdiscr;
		"""
	
	normal_code = "*p_normal = pos * invR;"

class CylinderComponent(ConvexIntersection.Component):
	"""Infinite cylinder"""
	
	extra_tracer_argument_definitions = ['const float3 axis', 'const float R2']
	extra_normal_argument_definitions = ['const float3 axis', 'const float invR']
	
	def __init__(self, axis, R):
		ConvexIntersection.Component.__init__(self)
		self.uax = normalize_tuple(axis)
		self.R = R
		
		self.extra_tracer_arguments = ["(float3)%s" % (self.uax,), self.R**2]
		self.extra_normal_arguments = ["(float3)%s" % (self.uax,), 1.0 / self.R]
	
	n_subobjects = 1
	
	tracer_code = """
	
		float z0 = dot(origin,axis), zslope = dot(ray,axis);
		
		float3 perp = origin - z0*axis;
		float3 ray_perp = ray - zslope*axis;
		
		float dotp = dot(ray_perp,perp);
		
		float perp2 = dot(perp,perp);
		float ray_perp2 = dot(ray_perp,ray_perp);
		
		float discr = dotp*dotp - ray_perp2*(perp2 - R2);
		
		if (discr < 0)
		{
			// ray does not hit the infinite cylinder
			*p_isec_begin = 1;
			*p_isec_end = 0;
			return;
		}
		
		// ray hits the infinite cylinder
		
		float sqrtdiscr = native_sqrt(discr);
		float d1 = -dotp - sqrtdiscr;
		
		*p_isec_begin = d1 / ray_perp2;
		*p_isec_end = (d1 + 2*sqrtdiscr) / ray_perp2;
		"""
	
	normal_code = """
		float3 perp = pos - dot(pos,axis)*axis;
		*p_normal = perp * invR;
		"""

class ConeComponent(ConvexIntersection.Component):
	"""Infinte cone"""
	
	extra_tracer_argument_definitions = [
		"const float3 axis",
		"const float s2"]
		
	extra_normal_argument_definitions = [
		"const float3 axis",
		"const float slope"]
	
	def __init__(self, pos, axis, slope):
		ConvexIntersection.Component.__init__(self,pos)
		self.axis = normalize_tuple(axis)
		self.slope = slope
		
		self.extra_tracer_arguments = [
			"(float3)%s" % (self.axis,),
			self.slope**2 ]
		
		self.extra_normal_arguments = [
			"(float3)%s" % (self.axis,),
			self.slope ]
	
	n_subobjects = 1
	
	tracer_code = """
		float z0 = dot(origin,axis);
		float ray_par_len = dot(ray,axis);
		
		float3 rel_par = z0*axis;
		float3 rel_perp = origin - rel_par;
		float3 ray_par = ray_par_len*axis;
		float3 ray_perp = ray - ray_par;
		float ray_perp_len2 = dot(ray_perp,ray_perp);
		float rel_perp_len2 = dot(rel_perp,rel_perp);
		
		float a = ray_perp_len2 - s2*ray_par_len*ray_par_len;
		float hb = dot(rel_perp,ray_perp) - s2 * ray_par_len*z0;
		float c = rel_perp_len2 - s2 * z0*z0;
		
		float discr = hb*hb - a*c;
		if (discr < 0) 
		{
			// ray does not hit the infinite cone
			*p_isec_begin = 1;
			*p_isec_end = 0;
			return;
		}
		
		float sqrtdiscr = native_sqrt(discr);
		float dist1, dist2, dist =  (-hb - sqrtdiscr)/a;
		
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
		
		float z1 = z0 + ray_par_len * dist1;
		float z2 = z0 + ray_par_len * dist2;
		
		if (z1 < 0 && z2 < 0) {
			// ray does not hit the semi-infinite cone
			*p_isec_begin = 1;
			*p_isec_end = 0;
		}
		else {
			if (z1 < 0) *p_isec_begin = dist2;
			else if (z2 < 0) *p_isec_end = dist1;
			else {
				*p_isec_begin = dist1;
				*p_isec_end = dist2;
			}
		}
		"""
	
	normal_code = """
		float z0 = dot(pos,axis);
		float3 rel_perp = pos - z0*axis;
		
		*p_normal = fast_normalize(rel_perp / (slope * z0) - axis * slope); // TODO
		"""
