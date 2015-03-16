### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'const float3 center, const float R2')
    
        if (origin_self && !inside)
        {
            // convex body
            return;
        }
        
        float3 rel = center - origin;
        float dotp = dot(ray, rel);
        float psq = dot(rel, rel);
        
        float dist, discr, sqrdiscr;
        
        if (dotp <= 0 && !inside)
        {
            // ray travelling away from the center, not starting inside 
            // the sphere => no intersection
            return;
        }
        
        discr = dotp*dotp - psq + R2;
        if(discr < 0) return;
        
        sqrdiscr = native_sqrt(discr);
        
        if (inside) dist = dotp + sqrdiscr;
        else dist = dotp - sqrdiscr;
        
        if (dist <= 0) return;
        *p_new_isec_dist = dist;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'const float3 center, const float invR')
    
        *p_normal = (pos - center) * invR;
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.pos)}}, {{obj.R*obj.R}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.pos)}}, {{1.0/obj.R}});
### endmacro
