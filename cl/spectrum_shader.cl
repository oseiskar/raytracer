
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
        constant float3 *emission,
        constant float3 *diffuse,
        constant float3 *reflecivity,
        constant float3 *transparency,
        constant float *ior,
        constant float *dispersion,
        constant float3 *vs,
        // the random sample [0,1) that decides what to do
        float p,
        // 
        float dispersion_coeff,
        // a random unit vector and color mask
        constant float4 *rvec_and_color_mask)
{
    const int gid = get_global_id(0);
    uint id = surface_object_id[gid];
    
    const float3 rvec = rvec_and_color_mask[0].xyz;
    const float3 cmask = rvec_and_color_mask[1].xyz;
    
    float cur_col = dot(cmask,vs[inside[gid]]);
    float cur_prob = 0;
    float cur_mult = 1.0;
    float3 r = ray[gid];
    float3 n = normal[gid];
    
    const float alpha = cur_col;
    
    if (alpha > 0) cur_prob = 1.0-exp(-alpha*last_distance[gid]);
    
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
        //float d = (p/cur_prob)*last_distance[gid]; 
      
        pos[gid] -= (last_distance[gid]-d) * r;
        //last_distance[gid] = d;
        surface_object_id[gid] = 0; // not on any surface
        r = rvec;
        cur_mult /= cur_prob;
    }
    else
    {
        p -= cur_prob;
    
        // TODO: the emission is non-Lambertian at the moment
        img[gid] += emission[id]*intensity[gid]*cmask;
        
        cur_col = dot(cmask,diffuse[id]);
        cur_prob = cur_col;
        
        if (p < cur_prob)
        {
            cur_mult /= cur_prob;
           
            // --- Diffuse
            r = rvec;
                
            // Reflect (negation) to outside
            if (dot(n,r) < 0) r = -r;
            
            // Lambert's law
            cur_mult *= 2.0 * dot(n,r);
        }
        else
        {
            p -= cur_prob;
            
            cur_col = dot(cmask,reflecivity[id]);
            cur_prob = cur_col;
            
            if (p < cur_prob)
            {
                cur_mult /= cur_prob;
                
                // --- Reflection
                r -= 2*dot(r,n) * n;
            }
            else
            {
                p -= cur_prob;
                
                cur_col = dot(cmask,transparency[id]);
                cur_prob = cur_col;
                
                if (p < cur_prob)
                {
                    cur_mult /= cur_prob;
                    
                    // --- Refraction / Transparency
                    
                    float dotp = dot(r,n);
                    if (dotp > 0) { n = -n; dotp = -dotp; }
                            
                    float nfrac = 0;
                    
                    float other_ior = ior[id] * (1.0 + dispersion[id]*dispersion_coeff);
                    
                    if (inside[gid] == surface_object_id[gid]) // Leaving
                    {
                        nfrac = other_ior/ior[0];
                        inside[gid] = 0; // TODO: parent
                    }
                    else // going in
                    {
                        nfrac = ior[0]/other_ior;
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

                }
                else
                {
                    // --- Assumed absorption
                    cur_col = 0;
                }
            }
        }
    }
    
    ray[gid] = r;
    intensity[gid] *= cur_mult*cur_col;
}
