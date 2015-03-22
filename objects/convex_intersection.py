
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
    
    def make_functions( self, template_env ):
        funcs = Tracer.make_functions( self, template_env )
        for component in self.components:
            subfuncs = component.make_functions(template_env)
            funcs = dict( funcs.items() + subfuncs.items() )
        return funcs
    
    def __init__(self, origin, components):
        self.origin = origin
        self.components = components
        
        self.unique_tracer_id = str(id(self))
    
    # "view helpers"
    
    def component_tracer_call_params(self, component):
        return 'rel - (float3)%s, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside' \
            % (tuple(component.pos),)
    
    def component_normal_call_params(self, component, subobject_offset):
        return 'p - (float3)%s, subobject - %d, p_normal' \
             % (tuple(component.pos), subobject_offset)
