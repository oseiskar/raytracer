from objects import ConvexIntersection, HalfSpaceComponent, FixedConvexIntersection
from utils import normalize_tuple
from transformations import Affine, rotation_aligning_vectors

class Cone(FixedConvexIntersection):
    
    def __init__(self, tip, axis, height, R):
        self.axis = axis
        self.height = height
        self.R = R
        
        z = (0,0,1)
        
        components = [
            HalfSpaceComponent(normal=z, h=1.0),
            ConeComponent((0, 0, 0), axis=z, slope=1.0)
        ]
        FixedConvexIntersection.__init__(self, tip, components)
        
    def tracer_coordinate_system(self):
        rotation = Affine(linear=rotation_aligning_vectors((0,0,1), self.axis))
        scaling = Affine(scaling=(self.R, self.R, self.height))
        return rotation(scaling)

class ConeComponent(ConvexIntersection.Component):
    """Infinte cone"""
    
    def __init__(self, pos, axis, slope):
        ConvexIntersection.Component.__init__(self, pos)
        self.axis = normalize_tuple(axis)
        self.slope = slope
    
    n_subobjects = 1

    def parameter_declarations(self):
        return ['float3 axis', 'float slope']
