### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_component_function_base(obj, 'const float3 axis, const float R2')
    
        float z0 = dot(origin,axis), zslope = dot(ray,axis);
        
        float3 perp = origin - z0*axis;
        float3 ray_perp = ray - zslope*axis;
        
        float dotp = dot(ray_perp,perp);
        
        float perp2 = dot(perp,perp);
        float ray_perp2 = dot(ray_perp,ray_perp);
        
        float discr = dotp*dotp - ray_perp2*(perp2 - R2);
        
        if (discr < 0)
        {
            // ray does not hit the infinite cylinder
            *p_isec_begin = 1;
            *p_isec_end = 0;
            return;
        }
        
        // ray hits the infinite cylinder
        
        float sqrtdiscr = native_sqrt(discr);
        float d1 = -dotp - sqrtdiscr;
        
        *p_isec_begin = d1 / ray_perp2;
        *p_isec_end = (d1 + 2*sqrtdiscr) / ray_perp2;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'const float3 axis, const float invR')
    
        float3 perp = pos - dot(pos,axis)*axis;
        *p_normal = perp * invR;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.uax)}}, {{obj.R*obj.R}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.uax)}}, {{1.0 / obj.R}});
### endmacro
