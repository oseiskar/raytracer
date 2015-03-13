from tracer import Tracer
from utils import normalize_tuple, vec_norm
import numpy
import sys, math

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
