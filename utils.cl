
/* Static OpenCL code */

__kernel void fill_vec_broadcast(global float3 *a, constant float3 *v)
{
	const int gid = get_global_id(0);
	a[gid] = *v;
}

__kernel void mult_by_param_vec_vec(
		global const uint *value_array,
		global float3 *target,
		constant float3 *param)
{
	const int gid = get_global_id(0);
	target[gid] *= param[value_array[gid]];
}

__kernel void subsample_transform_camera(
		global const float3 *original_rays,
		global float3 *new_rays,
		constant float3 *rotmat_rows)
{
	const int gid = get_global_id(0);
	
	const float3 ray = original_rays[gid];
	new_rays[gid] = (float3)(
	    dot(rotmat_rows[0],ray),
	    dot(rotmat_rows[1],ray),
	    dot(rotmat_rows[2],ray));
}

__kernel void prob_select_ray(
		global const uint *value_array,
		global float3 *normal,
		global float3 *ray,
		global float3 *color,
		global uint *inside,
		constant float3 *diffuse,
		constant float3 *reflecivity,
		constant float3 *transparency,
		constant float *ior,
		float p,
		constant float3 *rvec)
{
	const int gid = get_global_id(0);
	uint id = value_array[gid];
	
	float3 cur_col = diffuse[id];
	float cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	float cur_mult = 1.0;
	float3 r = ray[gid];
	float3 n = normal[gid];
	
	if (p < cur_prob)
	{
	    cur_mult /= cur_prob;
	    
	    // Diffuse
	    r = *rvec;
	    
	    // Reflect (negation) to outside
	    if (dot(n,r) < 0) r = -r;
	}
	else
	{
	    p -= cur_prob;
	    cur_mult /= (1.0 - cur_prob);
	    
	    cur_col = reflecivity[id];
	    cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	    
	    if (p < cur_prob)
	    {
	        cur_mult /= cur_prob;
	        
	        // Reflection
	        r -= 2*dot(r,n) * n;
	    }
	    else
	    {
	        p -= cur_prob;
	        cur_mult /= (1.0 - cur_prob);
	        
	        cur_col = transparency[id];
	        cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	        
	        if (p < cur_prob)
	        {
	            cur_mult /= cur_prob;
	            
	            // Refraction / Transparency
	            
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
	            // else leave r intact
	            
	            if (dot(n,r) < 0) n = -n;
	            normal[gid] = n;

	        }
	        else
	        {
	            // Assumed absorption
	            cur_col = 0;
	        }
	    }
	}
	
	ray[gid] = r;
    color[gid] *= cur_mult*cur_col;
}

