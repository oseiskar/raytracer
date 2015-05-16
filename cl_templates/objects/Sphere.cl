### extends 'tracer.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
        
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
        
        discr = dotp*dotp - psq + R*R;
        if(discr < 0) return;
        
        sqrdiscr = native_sqrt(discr);
        
        if (inside) dist = dotp + sqrdiscr;
        else dist = dotp - sqrdiscr;
        
        if (dist <= 0) return;
        *p_new_isec_dist = dist;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        *p_normal = (pos - center) / R;
        
    ### endcall
### endmacro
