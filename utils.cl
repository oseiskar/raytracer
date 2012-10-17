
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

__kernel void prob_select_ray(
		global const uint *value_array,
		global float3 *normal,
		global float3 *ray,
		global float3 *color,
		global uint *inside,
		constant float3 *diffuse,
		constant float3 *reflecivity,
		constant float3 *transparency,
		float p,
		constant float3 *rvec) // TODO
{
	const int gid = get_global_id(0);
	float3 c = color[gid];
	uint id = value_array[gid];
	
	float3 cur_col = diffuse[id];
	float cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	float cur_mult = 1.0;
	float3 r = ray[gid];
	
	if (p < cur_prob)
	{
	    // Diffuse
	    c *= cur_col/cur_prob;
	    r = *rvec;
	    
	    // Reflect (negation) to outside
	    if (dot(normal[gid],r) < 0) r = -r;
	}
	else
	{
	    p -= cur_prob;
	    cur_mult /= (1.0 - cur_prob);
	    
	    cur_col = reflecivity[id];
	    cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	    
	    if (p < cur_prob)
	    {
	        // Reflection
	        c *= cur_col/cur_prob;
	        r -= 2*dot(r,normal[gid]) * normal[gid];
	    }
	    else
	    {
	        p -= cur_prob;
	        cur_mult /= (1.0 - cur_prob);
	        
	        cur_col = transparency[id];
	        cur_prob = (cur_col.x+cur_col.y+cur_col.z)/3;
	        
	        if (p < cur_prob)
	        {
	            // Refraction
	            
	            // Transparency
	            // leave r intact
	            
	            c *= cur_col/cur_prob;
	        }
	        else
	        {
	            // Assumed absorption
	            c *= 0;
	        }
	    }
	}
	
	ray[gid] = r;
    color[gid] = c;
}

