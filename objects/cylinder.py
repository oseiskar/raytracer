from objects import ConvexIntersection, LayerComponent, FixedConvexIntersection
from utils import normalize_tuple
from transformations import Affine, rotation_aligning_vectors

class Cylinder(FixedConvexIntersection):
    
    def __init__(self, bottom_center, axis, height, R):
        
        self.axis = axis
        self.height = height
        self.R = R
        
        z = (0,0,1)
        
        components = [LayerComponent(z, 1.0), CylinderComponent(z, 1.0)]
        FixedConvexIntersection.__init__(self, bottom_center, components)

    def tracer_coordinate_system(self):
        rotation = Affine(linear=rotation_aligning_vectors((0,0,1), self.axis))
        scaling = Affine(scaling=(self.R, self.R, self.height))
        return rotation(scaling)

class CylinderComponent(ConvexIntersection.Component):
    """Infinite cylinder"""
    
    def __init__(self, axis, R):
        ConvexIntersection.Component.__init__(self)
        self.axis = normalize_tuple(axis)
        self.R = R
    
    n_subobjects = 1
    
    def parameter_declarations(self):
        return ['float3 axis', 'float R']
