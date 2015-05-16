### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)

        float slope = dot(ray,normal), d1, d2;
        
        if (slope > 0) {
            d1 = dot(-origin, normal);
            d2 = d1 + h;
        }
        else {
            d2 = dot(-origin, normal);
            d1 = d2 + h;
        }
        
        if ( (slope > 0) == inside ) *p_subobject = 1;
        else *p_subobject = 0;
        
        *p_isec_begin = d1 / slope;
        *p_isec_end = d2 / slope;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
        
        if (subobject == 1) *p_normal = normal;
        else *p_normal = -normal;
        
    ### endcall
### endmacro
