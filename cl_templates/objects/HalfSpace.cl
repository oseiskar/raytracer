### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
        if (!origin_self)
        {
            float slope = dot(ray,-normal);
            float dist = dot(origin, normal)+h;
            
            dist = dist/slope;
            if (dist > 0) *p_new_isec_dist = dist;
        }
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        *p_normal = normal;
        
    ### endcall
### endmacro
