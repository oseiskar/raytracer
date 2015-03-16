from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm
import numpy
import math

class Sphere(Tracer):
    
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

class SphereComponent(ConvexIntersection.Component):
    """Sphere"""
    
    n_subobjects = 1
    
    def __init__(self, pos, R):
        ConvexIntersection.Component.__init__(self,pos)
        self.R = R
    
