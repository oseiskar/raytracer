
from tracer import Tracer
import numpy

class ConvexIntersection(Tracer):
    """Intersection of convex objects represented by Components"""
    
    class Component(Tracer):
        
        def __init__( self, pos = (0, 0, 0) ):
            Tracer.__init__(self, position=pos)
    
    # freeze template name
    def template_name(self):
        return 'ConvexIntersection'
    
    def make_functions( self, template_env ):
        funcs = Tracer.make_functions( self, template_env )
        for component in self.components:
            subfuncs = component.make_functions(template_env)
            funcs = dict( funcs.items() + subfuncs.items() )
        return funcs
    
    def __init__(self, origin, components):
        Tracer.__init__(self, position=origin)
        self.components = components
        
        self.unique_tracer_id = '_' + '_'.join([c.__class__.__name__ \
            for c in self.components])
    
    # "view helpers"
    
    def component_tracer_call_params(self, component_idx):
        return ('rel - _pos_%i, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside' \
            % (component_idx,)) \
            + self.component_parameter_string(component_idx)
    
    def component_normal_call_params(self, component_idx, subobject_offset):
        component = self.components[component_idx]
        return ('p - _pos_%i, subobject - %d, p_normal' \
             % (component_idx, subobject_offset)) \
             + self.component_parameter_string(component_idx)
    
    def component_parameter_string(self, component_idx):
        params = []
        for name in self.components[component_idx].parameter_names():
            params.append('%s_%d' % (name, component_idx))
        
        return ', '.join([''] + params)

    @property
    def convex(self):
        return True
        
    def parameter_declarations(self):
        params = []
        for i in range(len(self.components)):
            c = self.components[i]
            params.append('float3 _pos_%d' % i)
            params += ['%s_%d' % (n,i) for n in c.parameter_declarations()]
        return params
    
    def parameter_values(self):
        params = []
        for c in self.components:
            params.append(c.position)
            params += c.parameter_values()
        return params

class UniqueConvexIntersection(ConvexIntersection):
    
    def __init__(self, origin, components):
        ConvexIntersection.__init__(self, origin, components)
        self.unique_tracer_id = str(id(self))
        
    def parameter_declarations(self):
        return []
    
    def parameter_values(self):
        return []
    
    # "view helpers"
    
    def _component_position_string(self, component_idx):
        return self._vec_literal(self.components[component_idx].position)
        
    def _vec_literal(self, vec):
        return "(float3)%s" % str(tuple(vec))
    
    def component_tracer_call_params(self, component_idx):
        return ('rel - %s, ray, &cur_ibegin, &cur_iend, &cur_subobj, inside' \
            % self._component_position_string(component_idx)) \
            + self.component_parameter_string(component_idx)
    
    def component_normal_call_params(self, component_idx, subobject_offset):
        component = self.components[component_idx]
        return ('p - %s, subobject - %d, p_normal' \
             % (self._component_position_string(component_idx), subobject_offset)) \
             + self.component_parameter_string(component_idx)
    
    def component_parameter_string(self, component_idx):
        params = []
        
        for val in self.components[component_idx].parameter_values():
            if len(numpy.ravel(val)) > 1:
                p = self._vec_literal(val)
            else:
                p = str(val)
            params.append(p)
        
        return ', '.join([''] + params)

class FixedConvexIntersection(UniqueConvexIntersection):
    def __init__(self, *args, **kwargs):
        UniqueConvexIntersection.__init__(self, *args, **kwargs)
        self.unique_tracer_id = ''
