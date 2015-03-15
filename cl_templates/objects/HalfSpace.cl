### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'const float3 normal, const float h')
    
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
    ### call normal_function_base(obj, 'const float3 normal')
    
        *p_normal = normal;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.normal)}}, {{obj.h}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.normal)}});
### endmacro
