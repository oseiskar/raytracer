from objects import ConvexIntersection, HalfSpaceComponent
from utils import normalize_tuple

class Cone(ConvexIntersection):
    
    def __init__(self, tip, axis, height, R):
        components = [
            HalfSpaceComponent(axis, height),
            ConeComponent( (0, 0, 0), axis, R / float(height))
        ]
        ConvexIntersection.__init__(self, tip, components)
        self.unique_tracer_id = ''

class ConeComponent(ConvexIntersection.Component):
    """Infinte cone"""
    
    def __init__(self, pos, axis, slope):
        ConvexIntersection.Component.__init__(self, pos)
        self.axis = normalize_tuple(axis)
        self.slope = slope
    
    n_subobjects = 1
