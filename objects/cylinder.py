from objects import ConvexIntersection, LayerComponent
from utils import normalize_tuple

class Cylinder(ConvexIntersection):
    
    def __init__(self, bottom_center, axis, height, R):
        components = [ LayerComponent(axis, height), CylinderComponent(axis,R) ]
        ConvexIntersection.__init__(self, bottom_center, components)
        self.unique_tracer_id = ''

class CylinderComponent(ConvexIntersection.Component):
    """Infinite cylinder"""
    
    def __init__(self, axis, R):
        ConvexIntersection.Component.__init__(self)
        self.uax = normalize_tuple(axis)
        self.R = R
    
    n_subobjects = 1
