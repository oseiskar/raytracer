
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


