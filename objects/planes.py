from tracer import Tracer
from objects import ConvexIntersection
from utils import normalize_tuple, vec_norm
from transformations import Affine, rotation_aligning_vectors
import numpy

class HalfSpace(Tracer):
    
    def __init__(self, normal, h):
        Tracer.__init__(self)
        self.normal = normalize_tuple(normal)
        self.h = h
    
    def tracer_coordinate_system(self):
        return Affine(
            linear = rotation_aligning_vectors((1,0,0), self.normal),
            translation = -numpy.ravel(self.normal)*self.h )
        
    @property
    def convex(self):
        return True

class HalfSpaceComponent(ConvexIntersection.Component):
    """Half-space"""
    
    def __init__(self, normal, h):
        ConvexIntersection.Component.__init__(self)
        
        self.normal = normalize_tuple(normal)
        self.h = h
        
    def parameter_declarations(self):
        return ['float3 normal', 'float h']
    
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

class ZLayerComponent(LayerComponent):
    def __init__(self):
        LayerComponent.__init__(self, (0,0,1), 1.0)

    def parameter_declarations(self):
        return []



