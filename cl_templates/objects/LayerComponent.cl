### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj, 'const float3 uax, const float h')

        float slope = dot(ray,uax), d1, d2;
        
        if (slope > 0) {
            d1 = dot(-origin, uax);
            d2 = d1 + h;
        }
        else {
            d2 = dot(-origin, uax);
            d1 = d2 + h;
        }
        
        if ( (slope > 0) == inside ) *p_subobject = 1;
        else *p_subobject = 0;
        
        *p_isec_begin = d1 / slope;
        *p_isec_end = d2 / slope;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'const float3 uax')
        
        if (subobject == 1) *p_normal = uax;
        else *p_normal = -uax;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.uax)}}, {{obj.h}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.uax)}});
### endmacro
