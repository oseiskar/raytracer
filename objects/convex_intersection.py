
from tracer import Tracer
from utils import normalize_tuple, vec_norm
import numpy

class ConvexIntersection(Tracer):
    """Intersection of convex objects represented by Components"""
    
    class Component(Tracer):
        
        def __init__( self, pos = (0,0,0) ):
            self.pos = tuple(pos)
    
    # freeze template name
    def template_name(self): return 'ConvexIntersection'
    
    def make_functions( self ):
        funcs = Tracer.make_functions( self )
        for component in self.components:
            subfuncs = component.make_functions()
            funcs = dict( funcs.items() + subfuncs.items() )
        return funcs
    
    def __init__(self, origin, components):
        self.origin = origin
        self.components = components
        
        self.unique_tracer_id = str(id(self))
    
    # "view helpers"
    
    def make_component_tracer_call(self, component):
        return component.make_tracer_call( \
            'rel - (float3)%s, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside' \
            % (tuple(component.pos),))
    
    def make_component_normal_call(self, component, subobject_offset):
        return component.make_normal_call( \
            'p - (float3)%s, subobject - %d, p_normal' \
             % (tuple(component.pos), subobject_offset))
