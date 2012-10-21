// runtime defines:
//      RUSSIAN_ROULETTE_PROB

__kernel void prob_select_ray(
		global float3 *img,
		global uint *value_array,
		global float3 *normal,
		global float *last_distance,
		global float3 *pos,
		global float3 *ray,
		global float3 *color,
		global uint *inside,
		constant float3 *emission,
		constant float3 *diffuse,
		constant float3 *reflecivity,
		constant float3 *transparency,
		constant float *ior,
		constant float3 *vs,
		float p,
		constant float3 *rvec,
		uint depth)
{
	const int gid = get_global_id(0);
	uint id = value_array[gid];
	
	float3 cur_col = vs[inside[gid]];
	float cur_prob = 0;
	float cur_mult = 1.0;
	float3 r = ray[gid];
	float3 n = normal[gid];
	
	const float alpha = (cur_col.x+cur_col.y+cur_col.z)/3 * 3.0;
	
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
	    value_array[gid] = 0; // not on any surface
	    r = *rvec;
	    cur_col /= alpha;
	}
	else
	{
	    img[gid] += emission[id]*color[gid];
	    // no reweighting here
	    
	    //p -= cur_prob;
	    //cur_mult /= (1.0 - cur_prob);
	    //cur_mult /= (1.0 - cur_prob);
	    
	    cur_col = diffuse[id];
	    cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	
	    if (p < cur_prob)
	    {
	        cur_mult /= cur_prob;
	       
	        // --- Diffuse
	        r = *rvec;
	            
	        // Reflect (negation) to outside
	        if (dot(n,r) < 0) r = -r;
	        
	        //if (depth == 1)
	        {
	            cur_mult *= 2.0 * dot(n,r);
	        }
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
	                
	                // TODO: fix this bias somewhere else...
	                cur_mult /= 1.0 - RUSSIAN_ROULETTE_PROB;
	                
	                // --- Refraction / Transparency
	                
	                float dotp = dot(r,n);
	                if (dotp > 0) { n = -n; dotp = -dotp; }
	                        
	                float nfrac = 0;
	                
	                if (inside[gid] == value_array[gid]) // Leaving
	                {
	                    nfrac = ior[id]/ior[0];
	                    inside[gid] = 0; // TODO: parent
	                }
	                else // going in
	                {
	                    nfrac = ior[0]/ior[id];
	                    inside[gid] = value_array[gid];
	                }
	                
	                if (nfrac != 1)
	                {
	                    float cos2t =1-nfrac*nfrac*(1-dotp*dotp);
	                    
	                    if (cos2t < 0)
	                    {
	                        // Total reflection
	                        
	                        r -= 2*dotp * n;
	                        
	                        cur_col = (float3)(1,1,1);
	                        cur_mult *= cur_prob;
	                        
	                        if (inside[gid] == value_array[gid]) inside[gid] = 0;
	                        else inside[gid] = value_array[gid];
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
