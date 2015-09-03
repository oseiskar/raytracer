from tracer import Tracer
from objects import ConvexIntersection
import numpy
from transformations import Affine

class Sphere(Tracer):
    
    def __init__(self, pos, R):
        Tracer.__init__(self, position=pos)
        self.R = R
    
    def surface_area(self):
        self._check_no_distortion()
        return 4.0 * numpy.pi * self.R**2

    def random_surface_point_and_normal(self):
        self._check_no_distortion()
        p = numpy.array(self.position)
        rand = numpy.random.normal(0, 1, p.shape)
        rand = rand / numpy.linalg.norm(rand)
        return (p + rand * self.R, rand)
    
    def center_and_min_sampling_distance(self):
        self._check_no_distortion()
        return (numpy.array(self.position), self.R * 2.0)
    
    def tracer_coordinate_system(self):
        return Affine(scaling=self.R)
    
    @property
    def convex(self):
        return True
    
    def _check_no_distortion(self):
        if not self.coordinates.is_orthogonal():
            raise RuntimeError("illegal operation for scaled sphere")
    

class SphereComponent(ConvexIntersection.Component):
    """Sphere"""
    
    n_subobjects = 1
    
    def __init__(self, pos, R):
        ConvexIntersection.Component.__init__(self, pos)
        self.R = R
        
    def parameter_declarations(self):
        return ['float R']
    
