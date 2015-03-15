from objects import ConvexIntersection, HalfSpaceComponent
from utils import normalize_tuple

class Cone(ConvexIntersection):
    
    def __init__(self, tip, axis, height, R):
        components = [ HalfSpaceComponent(axis, height), ConeComponent( (0,0,0), axis, R / float(height)) ]
        ConvexIntersection.__init__(self, tip, components)
        self.unique_tracer_id = ''

class ConeComponent(ConvexIntersection.Component):
    """Infinte cone"""
    
    extra_tracer_argument_definitions = [
        "const float3 axis",
        "const float s2"]
        
    extra_normal_argument_definitions = [
        "const float3 axis",
        "const float slope"]
    
    def __init__(self, pos, axis, slope):
        ConvexIntersection.Component.__init__(self,pos)
        self.axis = normalize_tuple(axis)
        self.slope = slope
        
        self.extra_tracer_arguments = [
            "(float3)%s" % (tuple(self.axis),),
            self.slope**2 ]
        
        self.extra_normal_arguments = [
            "(float3)%s" % (tuple(self.axis),),
            self.slope ]
    
    n_subobjects = 1
    
    tracer_code = """
        float z0 = dot(origin,axis);
        float ray_par_len = dot(ray,axis);
        
        float3 rel_par = z0*axis;
        float3 rel_perp = origin - rel_par;
        float3 ray_par = ray_par_len*axis;
        float3 ray_perp = ray - ray_par;
        float ray_perp_len2 = dot(ray_perp,ray_perp);
        float rel_perp_len2 = dot(rel_perp,rel_perp);
        
        float a = ray_perp_len2 - s2*ray_par_len*ray_par_len;
        float hb = dot(rel_perp,ray_perp) - s2 * ray_par_len*z0;
        float c = rel_perp_len2 - s2 * z0*z0;
        
        float discr = hb*hb - a*c;
        if (discr < 0) 
        {
            // ray does not hit the infinite cone
            *p_isec_begin = 1;
            *p_isec_end = 0;
            return;
        }
        
        float sqrtdiscr = native_sqrt(discr);
        float dist1, dist2, dist =  (-hb - sqrtdiscr)/a;
        
        if (a >= 0)
        {
            dist1 = dist;
            dist2 = dist + 2*sqrtdiscr/a;
        }
        else
        {
            dist2 = dist;
            dist1 = dist + 2*sqrtdiscr/a;
        }
        
        float z1 = z0 + ray_par_len * dist1;
        float z2 = z0 + ray_par_len * dist2;
        
        if (z1 < 0 && z2 < 0) {
            // ray does not hit the semi-infinite cone
            *p_isec_begin = 1;
            *p_isec_end = 0;
        }
        else {
            if (z1 < 0) *p_isec_begin = dist2;
            else if (z2 < 0) *p_isec_end = dist1;
            else {
                *p_isec_begin = dist1;
                *p_isec_end = dist2;
            }
        }
        """
    
    normal_code = """
        float z0 = dot(pos,axis);
        float3 rel_perp = pos - z0*axis;
        
        *p_normal = fast_normalize(rel_perp / (slope * z0) - axis * slope); // TODO
        """
