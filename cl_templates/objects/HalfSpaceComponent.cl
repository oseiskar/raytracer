### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj, 'const float3 normal, const float h')
    
        float slope = dot(ray,normal);
        float dist = dot(-origin,normal)+h;
        
        dist = dist/slope;
        if (slope < 0) *p_isec_begin = dist;
        else *p_isec_end = dist;
        *p_subobject = 0;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'const float3 normal')
        
        *p_normal = normal;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.normal_vec)}}, {{obj.h}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.normal_vec)}});
### endmacro
