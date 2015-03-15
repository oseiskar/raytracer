### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj, 'const float R2')
    
        float dotp = -dot(ray, origin);
        float psq = dot(origin, origin);
        
        float discr, sqrdiscr;
        
        discr = dotp*dotp - psq + R2;
        
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
    ### call normal_function_base(obj, 'const float invR')
    
        *p_normal = pos * invR;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{obj.R*obj.R}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{1.0/obj.R}});
### endmacro
