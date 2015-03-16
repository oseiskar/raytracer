### from 'helpers.cl' import vec3

### macro sphere_bounding_volume(center, R, minvar, maxvar)

    {
        // Bounding sphere intersection
        
        const float R2 = {{ R*R }};
        const float3 center = {{ vec3(center) }};
        float3 rel = center - origin;
        float dotp = dot(ray, rel);
        float psq = dot(rel, rel);
        
        bool inside_bnd = psq < R2;
        
        if (dotp <= 0 && !inside_bnd) return;
        
        const float discr = dotp*dotp - psq + R2;
        if(discr < 0) return;
        const float sqrdiscr = native_sqrt(discr);
        
        {{ minvar }} = max(dotp-sqrdiscr,0.0f);
        {{ maxvar }} = min(dotp+sqrdiscr,old_isec_dist);
        
        if ({{ maxvar }} <= {{ minvar }}) return;
        
    }
        
### endmacro
