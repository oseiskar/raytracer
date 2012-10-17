/*__kernel void saxpy(
		global float3 *out,
		global const float *alpha,
		global const float3 *x,
		global const float3 *y)
{
	const int gid = get_global_id(0);
	out[gid] = alpha[gid]*x[gid] + y[gid];
}*/

__kernel void normalize_vecs(global float3 *a)
{
	const int gid = get_global_id(0);
	a[gid] = normalize(a[gid]);
}

/*__kernel void fill_vec(global float3 *a, float x, float y, float z)
{
	const int gid = get_global_id(0);
	a[gid] = (float3)(x,y,z);
}*/

__kernel void fill_vec_broadcast(global float3 *a, constant float3 *v)
{
	const int gid = get_global_id(0);
	a[gid] = *v;
}

/*__kernel void update_dist_and_value(
		global const float *new_dist,
		global float *dist,
		global uint *value_array,
		uint value)
{
	const int gid = get_global_id(0);
	if (new_dist[gid] > 0 && new_dist[gid] < dist[gid])
	{
		dist[gid] = new_dist[gid];
		value_array[gid] = value;
	}
}*/

__kernel void cull(
		global const float3 *normal,
		global const float3 *ray,
		global float *target)
{
	const int gid = get_global_id(0);
	if (dot(normal[gid],ray[gid]) <= 0) target[gid] = 0;
}

/*__kernel void reflect(
        global float3 *ray,
		global const float3 *normal)
{
	const int gid = get_global_id(0);
	ray[gid] -= 2*dot(ray[gid],normal[gid]) * normal[gid];
}*/

/*__kernel void reflect_if_wrong_side(
		global const float3 *normal,
		global float3 *ray)
{
	const int gid = get_global_id(0);
	if (dot(normal[gid],ray[gid]) < 0) ray[gid] = -ray[gid];
}*/

__kernel void shadow_cull(
		global const float *dist,
		global const float *threshold,
		global float *target)
{
	const int gid = get_global_id(0);
	if (dist[gid] > 0 && dist[gid] < threshold[gid]) target[gid] = 0;
}

/*__kernel void cull_self(
		global uint *value_array,
		global float *target,
		uint value)
{
	const int gid = get_global_id(0);
	if (value_array[gid] == value) target[gid] = 0;
}*/

__kernel void mult_by_param(
		global const uint *value_array,
		global float *target,
		constant float *param)
{
	const int gid = get_global_id(0);
	target[gid] *= param[value_array[gid]];
}

__kernel void mult_by_param_vec_vec(
		global const uint *value_array,
		global float3 *target,
		constant float3 *param)
{
	const int gid = get_global_id(0);
	target[gid] *= param[value_array[gid]];
}

/*__kernel void prob_select_and_mult2(
		global const uint *value_array,
		global float3 *target,
		global float3 *mul,
		global const float3 *src1,
		constant float3 *param_array0,
		constant float3 *param_array1,
		float p)
{
	const int gid = get_global_id(0);
	const float3 param1 = param_array1[value_array[gid]];
	const float prob1 = (param1.x+param1.y+param1.z)/3;
	if (p < prob1)
	{
	    mul[gid] *= param1/prob1;
	    target[gid] = src1[gid];
	}
	else
	{
	    const float3 param2 = param_array0[value_array[gid]];
	    mul[gid] *= param2/(1-prob1);
	}
}*/

__kernel void prob_select_ray(
		global const uint *value_array,
		global float3 *normal,
		global float3 *ray,
		global float3 *color,
		global uint *inside,
		constant float3 *diffuse,
		constant float3 *reflecivity,
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
	        // Assumed absorption
	        c *= 0;
	    }
	}
	
	ray[gid] = r;
    color[gid] = c;
}

__kernel void mult_by_param_vec_scalar(
		global const uint *value_array,
		global float3 *target,
		constant float *param)
{
	const int gid = get_global_id(0);
	target[gid] *= param[value_array[gid]];
}

/*__kernel void copy_vector_if_match(
		global const uint *value_array,
		global const float3 *src,
		global float3 *target,
		uint value)
{
	const int gid = get_global_id(0);
	if (value_array[gid] == value)
	{
		target[gid] = src[gid];
	}
}*/
