from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm
import numpy
import sys, math

class HalfSpace(Tracer):
    
    extra_normal_argument_definitions = ["const float3 normal"]
    extra_tracer_argument_definitions = ["const float3 normal", "const float h"]
    
    def __init__(self, normal, h):
        self.normal_vec = normalize_tuple(normal)
        self.h = h
        
        self.extra_tracer_arguments = ["(float3)%s" % (tuple(self.normal_vec),), self.h]
        self.extra_normal_arguments = ["(float3)%s" % (tuple(self.normal_vec),)]
    
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

class HalfSpaceComponent(ConvexIntersection.Component):
    """Half-space"""
    
    extra_tracer_argument_definitions = ["const float3 normal", "const float h"]
    extra_normal_argument_definitions = ["const float3 normal"]
    
    def __init__(self, normal, h):
        ConvexIntersection.Component.__init__(self)
        self.normal_vec = normalize_tuple(normal)
        self.h = h
        
        self.extra_tracer_arguments = ["(float3)%s" % (tuple(self.normal_vec),), self.h]
        self.extra_normal_arguments = ["(float3)%s" % (tuple(self.normal_vec),)]
    
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






