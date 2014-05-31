
# Tracer objects

from tracer import *
from utils import normalize_tuple, vec_norm
import numpy
import sys, math

from components import *

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

class Parallelepiped(ConvexIntersection):
	
	def __init__(self, origin, ax1, ax2, ax3):
		components = [ LayerComponent(ax) for ax in (ax1,ax2,ax3)]
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = ''

class Octahedron(ConvexIntersection):
	def __init__(self, origin, R):
		
		R *= 1 / 3.0
		sq_vertices = [(R,R), (R,-R), (-R,R), (-R,-R)]
		
		components = []
		for v in sq_vertices:
			cube_vertex = v + (R,)
			print cube_vertex
			layer = LayerComponent( tuple([-x for x in cube_vertex]), vec_norm(cube_vertex)*2.0 )
			layer.pos = cube_vertex
			components.append(layer)
		
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = ''
	

class Cylinder(ConvexIntersection):
	
	def __init__(self, bottom_center, axis, height, R):
		components = [ LayerComponent(axis, height), CylinderComponent(axis,R) ]
		ConvexIntersection.__init__(self, bottom_center, components)
		self.unique_tracer_id = ''

class Cone(ConvexIntersection):
	
	def __init__(self, tip, axis, height, R):
		components = [ HalfSpaceComponent(axis, height), ConeComponent( (0,0,0), axis, R / float(height)) ]
		ConvexIntersection.__init__(self, tip, components)
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

from implicits import *
