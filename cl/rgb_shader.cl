
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
        // material properties of the current object
        constant float3 *emission,
        constant float3 *diffuse,
        constant float3 *reflecivity,
        constant float3 *transparency,
        constant float *ior,
        constant float3 *vs,
        // the random sample [0,1) that decides what to do
        float p,
        // a random unit vector
        constant float3 *rvec)
{
    const int gid = get_global_id(0);
    uint id = surface_object_id[gid];
    
    float3 cur_col = vs[inside[gid]];
    float cur_prob = 0;
    float cur_mult = 1.0;
    float3 r = ray[gid];
    float3 n = normal[gid];
    
    const float alpha = (cur_col.x+cur_col.y+cur_col.z)/3;
    
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
        r = *rvec;
        cur_mult /= cur_prob;
    }
    else
    {
        p -= cur_prob;
        
        // TODO: the emission is non-Lambertian at the moment
        img[gid] += emission[id]*color[gid];
        
        cur_col = diffuse[id];
        cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
    
        if (p < cur_prob)
        {
            cur_mult /= cur_prob;
           
            // --- Diffuse
            r = *rvec;
                
            // Reflect (negation) to outside
            if (dot(n,r) < 0) r = -r;
            
            // Lambert's law
            cur_mult *= 2.0 * dot(n,r);
        }
        else
        {
            p -= cur_prob;
            
            cur_col = reflecivity[id];
            cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
            
            if (p < cur_prob)
            {
                cur_mult /= cur_prob;
                
                // --- Reflection
                r -= 2*dot(r,n) * n;
            }
            else
            {
                p -= cur_prob;
                
                cur_col = transparency[id];
                cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
                
                if (p < cur_prob)
                {
                    cur_mult /= cur_prob;
                    
                    // --- Refraction / Transparency
                    
                    float dotp = dot(r,n);
                    if (dotp > 0) { n = -n; dotp = -dotp; }
                            
                    float nfrac = 0;
                    
                    if (inside[gid] == surface_object_id[gid]) // Leaving
                    {
                        nfrac = ior[id]/ior[0];
                        inside[gid] = 0; // TODO: parent
                    }
                    else // going in
                    {
                        nfrac = ior[0]/ior[id];
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
    color[gid] *= cur_mult*cur_col;
}
