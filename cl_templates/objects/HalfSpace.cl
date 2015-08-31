### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
        const float dist = -origin.x/ray.x;
        if (dist > 0) *p_new_isec_dist = dist;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        *p_normal = (float3)(1.0, 0.0, 0.0);
        
    ### endcall
### endmacro
