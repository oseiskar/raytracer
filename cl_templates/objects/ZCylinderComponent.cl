### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)
        
        float2 perp = origin.xy;
        float2 ray_perp = ray.xy;
        
        float dotp = dot(ray_perp,perp);
        
        float perp2 = dot(perp,perp);
        float ray_perp2 = dot(ray_perp,ray_perp);
        
        float discr = dotp*dotp - ray_perp2*(perp2 - 1.0);
        
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
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        float3 perp = pos;
        perp.z = 0;
        *p_normal = perp;
        
    ### endcall
### endmacro
