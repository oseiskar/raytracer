from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm

class BasicHalfSpace(Tracer):
    
    def __init__(self, normal, h):
        self.normal = normalize_tuple(normal)
        self.h = h
    
    def parameter_declarations(self):
        return ['float3 normal', 'float h']
        
    @property
    def convex(self):
        return True

class HalfSpace(BasicHalfSpace):
    def __init__(self, normal, h):
        BasicHalfSpace.__init__(self,normal,h)

class HalfSpaceComponent(BasicHalfSpace, ConvexIntersection.Component):
    """Half-space"""
    
    def __init__(self, normal, h):
        ConvexIntersection.Component.__init__(self)
        BasicHalfSpace.__init__(self, normal, h)
    
    n_subobjects = 1

class LayerComponent(HalfSpaceComponent):
    """Infinite layer with finite thickness"""
    
    def __init__(self, axis, h = None):
        if h is None:
            h = vec_norm(axis)
        else:
            h = h
        HalfSpaceComponent.__init__(self, axis, h)
    
    n_subobjects = 2






