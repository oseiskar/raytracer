
### for item in shader.get_material_property_offsets()
    #define MAT_{{item[0]}} {{item[1]}}
### endfor

### if shader.rgb
    #define COLOR(prop, id) material_colors[prop + id].xyz
### else
    #define COLOR(prop, id) material_scalars[prop + id]
### endif

#define SCALAR(prop, id) material_scalars[prop + id]

__kernel void shader(
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
### if shader.rgb
        global float3 *color,
### else
        global float *intensity,
### endif
        // id of the object inside which the current ray position is,
        // 0 if in ambient space. (notice however, that the ambient
        // space also has fog and IoR material properties)
        global uint *inside,
### if renderer.bidirectional
        // visibility mask for light point
        global float *shadow_mask,
        global int *suppress_emission,
        // id of the light object
        int light_id,
        float light_area,
        float min_light_sampling_distance,
### endif
        // material properties of the current object
### if shader.rgb
        constant float4 *material_colors,
### endif
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
    
### if shader.rgb
    #define COLOR2PROB(color) dot(color,WHITE)/3.0
        
    const float3 WHITE = (float3)(1.0,1.0,1.0);
    float3 cur_col_mult = WHITE, cur_col;
### else
    #define COLOR2PROB(color) color

    float cur_col;
    const float3 cmask = rvecs_cmask_and_light[2].xyz;
### endif
    
    float last_dist = last_distance[gid];
    const float alpha = SCALAR(MAT_VOLUME_SCATTERING,inside[gid]);
    
    ### if renderer.bidirectional
        const uint suppressed_emission_id = suppress_emission[gid];
        suppress_emission[gid] = 0;
    ### endif
    
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
        
        ### if renderer.bidirectional
            suppress_emission[gid] = -1;
        ### endif
    }
    
    ### if shader.rgb
        cur_col = COLOR(MAT_VOLUME_ABSORPTION,inside[gid]);
        cur_col_mult = (float3)(exp(-cur_col.x*last_dist),exp(-cur_col.y*last_dist),exp(-cur_col.z*last_dist));
        cur_col = WHITE;
    ### else
        cur_mult *= exp(-SCALAR(MAT_VOLUME_ABSORPTION,inside[gid])*last_dist);
    ### endif
    
    if (p >= cur_prob)
    {
        p -= cur_prob;
        
        ### if renderer.bidirectional
            if (id != suppressed_emission_id)
            {
        ### endif
                img[gid] += COLOR(MAT_EMISSION,id)
                    ### if shader.rgb
                        * color[gid]*cur_col_mult;
                    ### else
                         * intensity[gid]*cmask*cur_mult;
                    ### endif
        ### if renderer.bidirectional
            }
        ### endif
        
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
                ### if shader.rgb
                    cur_mult /= cur_prob;
                ### endif
                
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
                ### if shader.rgb
                cur_col = COLOR(MAT_DIFFUSE,id);
                
                // diffusion / absorbtion
                
                if (COLOR2PROB(cur_col) > 0.0) {
                ### else
                cur_mult *= COLOR(MAT_DIFFUSE,id);
                
                if (cur_mult > 0.0) {
                ### endif
                
                    ### if renderer.bidirectional
                        if (suppressed_emission_id == 0 && light_id > 0) {
                            const float3 light_center = rvecs_cmask_and_light[5].xyz;
                            
                            if (length(pos[gid]-light_center) > min_light_sampling_distance) {
                            
                                const float3 light_point = rvecs_cmask_and_light[3].xyz;
                                float3 shadow_ray = light_point - pos[gid];
                                const float3 light_normal = rvecs_cmask_and_light[4].xyz;
                            
                                suppress_emission[gid] = light_id;
                                
                                if (dot(shadow_ray, n) > 0 && dot(shadow_ray,light_normal) < 0) {
                                    const float shadow_dist = length(shadow_ray);
                                    shadow_ray = fast_normalize(shadow_ray);
                                    
                                    img[gid] += dot(n,shadow_ray) / M_PI
                                        * dot(-shadow_ray,light_normal) / (shadow_dist*shadow_dist)
                                    ### if shader.rgb
                                        * color[gid]*cur_mult*cur_col*cur_col_mult
                                    ### else
                                        * intensity[gid]*cmask*cur_mult
                                    ### endif
                                        * COLOR(MAT_EMISSION,light_id)
                                        * shadow_mask[gid]
                                        * light_area;
                                }
                            }
                        }
                    ### endif
                
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
    
    ### if shader.rgb
        color[gid] *= cur_mult*cur_col*cur_col_mult;
    ### else
        intensity[gid] *= cur_mult;
    ### endif
}
