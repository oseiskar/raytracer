### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)
    
        float slope = dot(ray,normal);
        float dist = dot(-origin,normal)+h;
        
        dist = dist/slope;
        if (slope < 0) *p_isec_begin = dist;
        else *p_isec_end = dist;
        *p_subobject = 0;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
        
        *p_normal = normal;
        
    ### endcall
### endmacro
