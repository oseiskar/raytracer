
#define PROPERTY(prop, id) material_properties[prop + id]

__kernel void spectrum_shader(
        // output: colors are summed to this image when appropriate
        global float3 *img,
        // id of the object on whose surface the ray position (param
        // pos below) is currently on. If 0, the ray position is not
        // on any surface (happens with fog/scattering and rays
        // starting from the camera)
        global uint *surface_object_id,
        // surface normal at ray position (if on a surface)
        global float3 *normal,
        // ...
        global float *last_distance,
        // current ray position
        global float3 *pos,
        // current ray direction
        global float3 *ray,
        // current ray color: a filter that multiplies everything
        // added to the image
        global float *intensity,
        // id of the object inside which the current ray position is,
        // 0 if in ambient space. (notice however, that the ambient
        // space also has fog and IoR material properties)
        global uint *inside,
        // material properties of the current object
        constant float *material_properties,
        // the random sample [0,1) that decides what to do
        float p,
        // a random unit vector and color mask
        constant float4 *rvecs_and_cmask)
{
    const int gid = get_global_id(0);
    uint id = surface_object_id[gid];
    
    float3 r = ray[gid];
    float3 n = normal[gid];
    const float3 rvec = rvecs_and_cmask[0].xyz;
    float3 gauss_rvec = rvecs_and_cmask[1].xyz;
    
    float cur_prob = 0;
    float cur_mult = 1.0;
    float blur;
    
    const float3 cmask = rvecs_and_cmask[2].xyz;
    
    
    float last_dist = last_distance[gid];
    const float alpha = PROPERTY(MAT_VOLUME_SCATTERING,inside[gid]);
    
    if (alpha > 0) cur_prob = 1.0-exp(-alpha*last_dist);
    
    if (p < cur_prob)
    {
        // --- volume scattering
        
        // "derivation":
        //    p < 1.0-exp(-alpha*dist)
        //   (1-p) > exp(-alpha*dist)
        //   log(1-p) > -alpha*dist
        //   -log(1-p) < alpha*dist
        //   dist > -log(1-p)/alpha
        float d = -log(1.0 - p) / alpha;
      
        pos[gid] -= (last_dist-d) * r;
        last_dist = d;
        surface_object_id[gid] = 0; // not on any surface
        
        blur = PROPERTY(MAT_VOLUME_SCATTERING_BLUR,inside[gid]);
        if (blur < 1.0) {
            r = normalize(gauss_rvec + r * tan(M_PI*0.5*(1.0 - blur)));
        }
        else r = rvec;
    }
    
    cur_mult *= exp(-PROPERTY(MAT_VOLUME_ABSORPTION,inside[gid])*last_dist);
    
    if (p >= cur_prob)
    {
        p -= cur_prob;
    
        // TODO: the emission is non-Lambertian at the moment
        img[gid] += PROPERTY(MAT_EMISSION, id)*intensity[gid]*cmask*cur_mult;
        
        cur_prob = PROPERTY(MAT_REFLECTION,id);
        
        if (p < cur_prob)
        {
            // --- Reflection
            r -= 2*dot(r,n) * n;
            
            blur = PROPERTY(MAT_REFLECTION_BLUR,id);
            if (blur > 0) {
                if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
                r = normalize(gauss_rvec * blur + r * (1.0-blur));
            }
        }
        else
        {
            p -= cur_prob;
            
            cur_prob = PROPERTY(MAT_TRANSPARENCY,id);
            
            if (p < cur_prob)
            {
                // --- Refraction / Transparency
                
                float dotp = dot(r,n);
                if (dotp > 0) { n = -n; dotp = -dotp; }
                        
                float nfrac = 0;
                
                float ior1 = PROPERTY(MAT_IOR,id);
                float ior0 = PROPERTY(MAT_IOR,0);
                
                if (inside[gid] == surface_object_id[gid]) // Leaving
                {
                    nfrac = ior1/ior0;
                    inside[gid] = 0; // TODO: parent
                }
                else // going in
                {
                    nfrac = ior0/ior1;
                    inside[gid] = surface_object_id[gid];
                }
                
                if (nfrac != 1)
                {
                    float cos2t =1-nfrac*nfrac*(1-dotp*dotp);
                    
                    if (cos2t < 0)
                    {
                        // Total reflection
                        
                        r -= 2*dotp * n;
                        
                        if (inside[gid] == surface_object_id[gid]) inside[gid] = 0;
                        else inside[gid] = surface_object_id[gid];
                    }
                    else
                    {
                        // Refraction
                        r = normalize(r*nfrac + n*(-dotp*nfrac-sqrt(cos2t)));
                    }
                }
                
                // else: no refraction, leave r intact
                
                if (dot(n,r) < 0) n = -n;
                normal[gid] = n;
                
                blur = PROPERTY(MAT_TRANSPARENCY_BLUR,id);
                if (blur > 0) {
                    if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
                    r = normalize(gauss_rvec * blur + r * (1.0-blur));
                }

            }
            else
            {
                cur_mult *= PROPERTY(MAT_DIFFUSE,id);
                
                if (cur_mult > 0.0)
                {
                    // --- Diffuse
                    r = rvec;
                        
                    // Reflect (negation) to outside
                    if (dot(n,r) < 0) r = -r;
                    
                    // Lambert's law
                    cur_mult *= 2.0 * dot(n,r);
                }
            }
        }
    }
    
    ray[gid] = r;
    intensity[gid] *= cur_mult;
}
