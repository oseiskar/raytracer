
/* Static OpenCL code */

// Interval arithmetic
#define ia_type float2
#define ia_new_exact(ex) (ia_new(ex,ex))
#define ia_new(b,e) ((float2)(b,e))
#define ia_begin(a) ((a).x)
#define ia_end(a) ((a).y)
#define ia_len(a) (ia_end(a)-ia_begin(a))
#define ia_contains_zero(a) (ia_begin(a) <= 0 && ia_end(a) >= 0)
#define ia_center(a) ((ia_begin(a)+ia_end(a))*0.5)
#define ia_add(a, b) ((a)+(b))
#define ia_add_exact(a, ex) ((a)+(ex))
#define ia_iadd_exact(a, ex) a += ex
#define ia_iadd(a,b) a += b
#define ia_sub(a, b) ia_new(ia_begin(a)-ia_end(b), ia_end(a)-ia_begin(b))
#define ia_mul_pos_exact(a,ex) ((a)*(ex))
#define ia_mul_neg_exact(a,ex) ia_neg(ia_mul_pos_exact(a,-ex))
#define ia_mul_exact(a,ex) ((ex) > 0 ? ia_mul_pos_exact(a,ex) : ia_mul_neg_exact(a,ex))
#define ia_neg(a) (ia_new(-ia_end(a),-ia_begin(a)))
#define ia_abs(a) (ia_begin(a) >= 0 ? (a) : \
                   (ia_end(a) <= 0 ? ia_neg(a) : \
                    ia_new(0, max(-ia_begin(a),ia_end(a)))))

ia_type ia_mul(ia_type a, ia_type b)
{
    ia_type i1 = a*b;
    ia_type i2 = -ia_neg(a)*b;
    return ia_new(
        min(min(ia_begin(i1),ia_end(i1)),min(ia_begin(i2),ia_end(i2))),
        max(max(ia_begin(i1),ia_end(i1)),max(ia_begin(i2),ia_end(i2))));
}

ia_type ia_pow2(ia_type a)
{
    ia_type ab = ia_abs(a);
    return ab*ab;
}

ia_type ia_pow3(ia_type a)
{
    return a*a*a;
}

ia_type ia_pow4(ia_type a)
{
    ia_type a2 = ia_pow2(a);
    return a2*a2;
}

// TODO: write more of these dynamically


// quaternion multiplication routines from
// http://users.cms.caltech.edu/~keenan/project_qjulia.html

float4 quaternion_mult( float4 q1, float4 q2 )
{
   float4 r;
   r.x = q1.x*q2.x - dot( q1.yzw, q2.yzw );
   r.yzw = q1.x*q2.yzw + q2.x*q1.yzw + cross( q1.yzw, q2.yzw );
   return r;
}

float4 quaternion_square( float4 q )
{
   float4 r;
   r.x = q.x*q.x - dot( q.yzw, q.yzw );
   r.yzw = 2*q.x*q.yzw;
   return r;
}


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


