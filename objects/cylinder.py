from objects import ConvexIntersection, LayerComponent
from utils import normalize_tuple

class Cylinder(ConvexIntersection):
    
    def __init__(self, bottom_center, axis, height, R):
        components = [ LayerComponent(axis, height), CylinderComponent(axis,R) ]
        ConvexIntersection.__init__(self, bottom_center, components)
        self.unique_tracer_id = ''

class CylinderComponent(ConvexIntersection.Component):
    """Infinite cylinder"""
    
    extra_tracer_argument_definitions = ['const float3 axis', 'const float R2']
    extra_normal_argument_definitions = ['const float3 axis', 'const float invR']
    
    def __init__(self, axis, R):
        ConvexIntersection.Component.__init__(self)
        self.uax = normalize_tuple(axis)
        self.R = R
        
        self.extra_tracer_arguments = ["(float3)%s" % (self.uax,), self.R**2]
        self.extra_normal_arguments = ["(float3)%s" % (self.uax,), 1.0 / self.R]
    
    n_subobjects = 1
    
    tracer_code = """
    
        float z0 = dot(origin,axis), zslope = dot(ray,axis);
        
        float3 perp = origin - z0*axis;
        float3 ray_perp = ray - zslope*axis;
        
        float dotp = dot(ray_perp,perp);
        
        float perp2 = dot(perp,perp);
        float ray_perp2 = dot(ray_perp,ray_perp);
        
        float discr = dotp*dotp - ray_perp2*(perp2 - R2);
        
        if (discr < 0)
        {
            // ray does not hit the infinite cylinder
            *p_isec_begin = 1;
            *p_isec_end = 0;
            return;
        }
        
        // ray hits the infinite cylinder
        
        float sqrtdiscr = native_sqrt(discr);
        float d1 = -dotp - sqrtdiscr;
        
        *p_isec_begin = d1 / ray_perp2;
        *p_isec_end = (d1 + 2*sqrtdiscr) / ray_perp2;
        """
    
    normal_code = """
        float3 perp = pos - dot(pos,axis)*axis;
        *p_normal = perp * invR;
        """
