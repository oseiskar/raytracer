### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj)

        float slope = ray.z, d1, d2;
        
        if (slope > 0) {
            d1 = -origin.z;
            d2 = d1 + 1.0;
        }
        else {
            d2 = -origin.z;
            d1 = d2 + 1.0;
        }
        
        if ( (slope > 0) == inside ) *p_subobject = 1;
        else *p_subobject = 0;
        
        *p_isec_begin = d1 / slope;
        *p_isec_end = d2 / slope;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
        
        if (subobject == 1) *p_normal = (float3)(0.0,0.0,1.0);
        else *p_normal = (float3)(0.0,0.0,-1.0);
        
    ### endcall
### endmacro
