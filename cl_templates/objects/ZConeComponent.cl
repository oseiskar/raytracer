### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)
    
        const float z0 = origin.z;
        const float ray_par_len = ray.z;
        
        const float2 rel_perp = origin.xy;
        const float2 ray_perp = ray.xy;
        const float ray_perp_len2 = dot(ray_perp,ray_perp);
        const float rel_perp_len2 = dot(rel_perp,rel_perp);
        
        const float a = ray_perp_len2 - ray_par_len*ray_par_len;
        const float hb = dot(rel_perp,ray_perp) - ray_par_len*z0;
        const float c = rel_perp_len2 - z0*z0;
        
        const float discr = hb*hb - a*c;
        if (discr < 0) 
        {
            // ray does not hit the infinite cone
            *p_isec_begin = 1;
            *p_isec_end = 0;
            return;
        }
        
        const float sqrtdiscr = native_sqrt(discr);
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
        
        const float z1 = z0 + ray_par_len * dist1;
        const float z2 = z0 + ray_par_len * dist2;
        
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
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        float3 nn = pos / pos.z;
        nn.z = -1.0;
        
        *p_normal = nn;
        
    ### endcall
### endmacro
