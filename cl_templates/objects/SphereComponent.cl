### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)
    
        float dotp = -dot(ray, origin);
        float psq = dot(origin, origin);
        
        float discr, sqrdiscr;
        
        discr = dotp*dotp - psq + R*R;
        
        if(discr < 0) {
            // ray does not hit the sphere
            *p_isec_begin = 1;
            *p_isec_end = 0;
            return;
        }
        
        sqrdiscr = native_sqrt(discr);
        
        *p_isec_begin = dotp - sqrdiscr;
        *p_isec_end = dotp + sqrdiscr;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        *p_normal = pos / R;
        
    ### endcall
### endmacro
