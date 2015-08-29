### extends 'tracer.cl'

### from 'helpers.cl' import vec3
### from 'objects/bounding_volumes.cl' import sphere_bounding_volume

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
        ### if obj.no_self_intersection
        if (origin_self) return;
        ### endif
        
        float t, t_end;
        float sign = 1.0;
        if (inside) sign = -1.0;
        
        ### if obj.bndR
            {{ sphere_bounding_volume(obj.center, obj.bndR, 't', 't_end') }};
        ### else
            t = 0.0;
            t_end = old_isec_dist;
        ### endif
        
        const int MAX_ITER = {{ obj.max_itr }};
        const float EPS = {{ obj.precision }};
        
        const float3 rel_origin = origin - {{ vec3(obj.center) }};
        int i;
        float x,y,z;
        
        float last_dist;
        float derivative = -1.0;
        
        // basic ray marching
        for( i=0; i < MAX_ITER; i++ )
        {
            if (t > t_end) return;
            
            float3 pos = rel_origin + t * ray;
            x = pos.x;
            y = pos.y;
            z = pos.z;
            
            float dist = 0.0;
            {
            {{ obj.tracer_code }};
            }
            dist *= sign;
                
            if (dist < EPS) {
                if (origin_self) {
                    if (i > 0) derivative = dist - last_dist;
                    else derivative = 0.0;
                }
                
                if (derivative < 0.0) {
                    if (t < t_end) *p_new_isec_dist = t;
                    return;
                }
            }
            
            t += max(dist, EPS);
            last_dist = dist;
        }
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
        const float3 rel_pos = pos - {{ vec3(obj.center) }};
        float3 normal;
        float x,y,z;
        
        ### if obj.normal_code
            x = rel_pos.x;
            y = rel_pos.y;
            z = rel_pos.z;
            {
            {{ obj.normal_code }}
            }
        ### else
            
            // default normal code
            const float EPS = {{ obj.precision }};
            
            float delta, dist;
            
            ### for var in ['x', 'y', 'z']
            
                delta = 0.0;
                
                ### for sign in [1, -1]
                    x = rel_pos.x;
                    y = rel_pos.y;
                    z = rel_pos.z;
                    {{ var }} += 0.5 * EPS * {{ sign }};
                    
                    {
                    {{ obj.tracer_code }};
                    }
                    delta += dist * {{ sign }};
                    
                ### endfor
                
                normal.{{ var }} = delta;
                
            ### endfor
            
        ### endif
        *p_normal = normalize(normal);
    ### endcall
### endmacro
