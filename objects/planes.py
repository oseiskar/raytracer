from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm

class HalfSpace(Tracer):
    
    def __init__(self, normal, h):
        self.normal = normalize_tuple(normal)
        self.h = h
    
    @property
    def convex(self):
        return True

class HalfSpaceComponent(ConvexIntersection.Component):
    """Half-space"""
    
    def __init__(self, normal, h):
        ConvexIntersection.Component.__init__(self)
        self.normal_vec = normalize_tuple(normal)
        self.h = h
    
    n_subobjects = 1

class LayerComponent(ConvexIntersection.Component):
    """Infinite layer with finite thickness"""
    
    def __init__(self, axis, h = None):
        ConvexIntersection.Component.__init__(self)
        self.uax = normalize_tuple(axis)
        if h == None:
            self.h = vec_norm(axis)
        else:
            self.h = h
    
    n_subobjects = 2






