### extends 'tracer.cl'
### from 'helpers.cl' import vec3

### from 'objects/bounding_volumes.cl' import sphere_bounding_volume

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
        ### if obj.no_self_intersection
        if (origin_self) return;
        ### endif
    
        ia_type t;
        
        ### if obj.bndR
            {{ sphere_bounding_volume(obj.center, obj.bndR*obj.scale, 'ia_begin(t)', 'ia_end(t)') }};
        ### else
            ia_begin(t) = 0.0;
            ia_end(t) = old_isec_dist;
        ### endif
        
        int i=0;
        const int MAX_ITER = {{ obj.max_itr }};
        const float TARGET_EPS = {{ obj.precision }};
        const float FRACTION = 0.5;
        const float SELF_MAX_BEGIN_STEP = 0.1;
        
        ia_type f, df;
        ia_type x,y,z;
        
        float step;
        int steps_since_subdiv = 0;
        
        const ia_type ray_x = ia_new_exact(ray.x);
        const ia_type ray_y = ia_new_exact(ray.y);
        const ia_type ray_z = ia_new_exact(ray.z);
        const ia_type origin_x = ia_new_exact(origin.x);
        const ia_type origin_y = ia_new_exact(origin.y);
        const ia_type origin_z = ia_new_exact(origin.z);
        int need_subdiv;
        
        if (origin_self)
        {
            ia_end(t) = min(ia_end(t),ia_begin(t) + SELF_MAX_BEGIN_STEP);
        }

        for( i=0; i < MAX_ITER; i++ )
        {
            if (ia_begin(t) >= old_isec_dist) return;
            if (ia_end(t) > old_isec_dist) ia_end(t) = old_isec_dist;
            
            x = ia_mul(ray_x,t) + origin_x;
            y = ia_mul(ray_y,t) + origin_y;
            z = ia_mul(ray_z,t) + origin_z;
        
        
            ### if obj.f_code_template()
                ### import obj.f_code_template() as t
                {{ t.f_code(obj) }}
            ### else
                {{ obj.f_code }}
            ### endif
            
            step = ia_len(t);
            need_subdiv = 0;
            
            if ((inside && ia_end(f) > 0) || (!inside && ia_begin(f) < 0))
            {
                need_subdiv = 1;
                
                if (origin_self)
                {
                        
                    ### if obj.f_code_template()
                        {{ t.df_code(obj) }}
                    ### else
                        {{ obj.df_code }}
                    ### endif
                    
                    if (!ia_contains_zero(df)) {
                        if ( (ia_end(df) < 0) == inside ) need_subdiv = 0;
                    }
                }
                
                if ( need_subdiv )
                {
                    if (step < TARGET_EPS || i == MAX_ITER-1)
                    {
                        step = ia_center(t);
                        *p_new_isec_dist = step;
                        return;
                    }
                
                    // Subdivide
                    step *= FRACTION;
                    ia_end(t) = ia_begin(t) + step;
                    steps_since_subdiv = 0;
                    continue;
                }
                
            }
            steps_since_subdiv++;
            
            // Step forward
            if (steps_since_subdiv > 1) step /= FRACTION;
            t = ia_new(ia_end(t),ia_end(t)+step);
        }
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        *p_normal = fast_normalize({{vec3(obj.gradient_code)}});
        
    ### endcall
### endmacro
