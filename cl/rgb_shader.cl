
#define COLOR(prop, id) material_colors[prop + id].xyz
#define SCALAR(prop, id) material_scalars[prop + id]

__kernel void rgb_shader(
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
        global float3 *color,
        // id of the object inside which the current ray position is,
        // 0 if in ambient space. (notice however, that the ambient
        // space also has fog and IoR material properties)
        global uint *inside,
#ifdef BIDIRECTIONAL
        // visibility mask for light point
        global float *shadow_mask,
        global int *suppress_emission,
        // id of the light object
        int light_id,
        float light_area,
#endif
        // material properties of the current object
        constant float4 *material_colors,
        constant float *material_scalars,
        // the random sample [0,1) that decides what to do
        float p,
        // a random unit vector and a random gaussian vector
        constant float4 *rvecs_cmask_and_light)
{
    const int gid = get_global_id(0);
    uint id = surface_object_id[gid];
    
    float3 r = ray[gid];
    float3 n = normal[gid];
    const float3 rvec = rvecs_cmask_and_light[0].xyz;
    float3 gauss_rvec = rvecs_cmask_and_light[1].xyz;
    
    float cur_prob = 0;
    float cur_mult = 1.0;
    float blur;
    
    const float3 WHITE = (float3)(1.0,1.0,1.0);
    float3 cur_col_mult = WHITE;
    
    float last_dist = last_distance[gid];
    const float alpha = SCALAR(MAT_VOLUME_SCATTERING,inside[gid]);
    
    #ifdef BIDIRECTIONAL
        const uint suppressed_emission_id = suppress_emission[gid];
        if (suppressed_emission_id != 0) {
            suppress_emission[gid] = -1;
        }
    #endif
    
    if (alpha > 0) cur_prob = 1.0-exp(-alpha*last_dist);
    
    if (p < cur_prob)
    {
        // --- (sub-surface) scattering / fog
        
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
        
        blur = SCALAR(MAT_VOLUME_SCATTERING_BLUR,inside[gid]);
        if (blur < 1.0) {
            r = normalize(gauss_rvec + r * tan(M_PI*0.5*(1.0 - blur)));
        }
        else r = rvec;
        
        #ifdef BIDIRECTIONAL
            suppress_emission[gid] = -1;
        #endif
    }
    
    #define COLOR2PROB(color) dot(color,WHITE)/3.0
    
    float3 cur_col = COLOR(MAT_VOLUME_ABSORPTION,inside[gid]);
    cur_col_mult = (float3)(exp(-cur_col.x*last_dist),exp(-cur_col.y*last_dist),exp(-cur_col.z*last_dist));
    cur_col = WHITE;
    
    if (p >= cur_prob)
    {
        p -= cur_prob;
        
        // TODO: the emission is non-Lambertian at the moment
        #ifdef BIDIRECTIONAL
            if (id != suppressed_emission_id)
            {
        #endif
                img[gid] += COLOR(MAT_EMISSION,id)*color[gid]*cur_col_mult;
        #ifdef BIDIRECTIONAL
            }
        #endif
        
        cur_col = COLOR(MAT_REFLECTION,id);
        cur_prob = COLOR2PROB(cur_col);
        
        if (p < cur_prob)
        {
            cur_mult /= cur_prob;
            
            // --- Reflection
            r -= 2*dot(r,n) * n;
            
            blur = SCALAR(MAT_REFLECTION_BLUR,id);
            if (blur > 0) {
                if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
                r = normalize(gauss_rvec * blur + r * (1.0-blur));
            }
        }
        else
        {
            p -= cur_prob;
            
            cur_col = COLOR(MAT_TRANSPARENCY,id);
            cur_prob = COLOR2PROB(cur_col);
            
            if (p < cur_prob)
            {
                cur_mult /= cur_prob;
                
                // --- Refraction / Transparency
                
                float dotp = dot(r,n);
                if (dotp > 0) { n = -n; dotp = -dotp; }
                        
                float nfrac = 0;
                
                float ior1 = SCALAR(MAT_IOR,id);
                float ior0 = SCALAR(MAT_IOR,0);
                
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
                
                blur = SCALAR(MAT_TRANSPARENCY_BLUR,id);
                if (blur > 0) {
                    if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
                    r = normalize(gauss_rvec * blur + r * (1.0-blur));
                }

            }
            else
            {
                cur_col = COLOR(MAT_DIFFUSE,id);
                
                // diffusion / absorbtion
                
                if (COLOR2PROB(cur_col) > 0.0) {
                
                    #ifdef BIDIRECTIONAL
                        if (suppressed_emission_id == 0) {
                            const float3 light_point = rvecs_cmask_and_light[3].xyz;
                            const float3 light_normal = rvecs_cmask_and_light[4].xyz;
                            float3 shadow_ray = light_point - pos[gid];
                            const float shadow_dist = length(shadow_ray);
                            
                            suppress_emission[gid] = light_id;
                            
                            if (dot(shadow_ray, n) > 0 && dot(shadow_ray,light_normal) < 0) {
                                shadow_ray = fast_normalize(shadow_ray);
                                
                                img[gid] += 2.0 * dot(n,shadow_ray)
                                    * dot(-shadow_ray,light_normal) / (shadow_dist*shadow_dist)
                                    * color[gid]*cur_mult*cur_col*cur_col_mult
                                    * COLOR(MAT_EMISSION,light_id)
                                    * shadow_mask[gid]
                                    * light_area;
                            }
                        }
                    #endif
                
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
    color[gid] *= cur_mult*cur_col*cur_col_mult;
}
