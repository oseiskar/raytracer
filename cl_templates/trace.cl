
### for i in range(n_objects)
    
__kernel void trace_object_{{ i }}(
    __global const float3 *p_pos,
    __global const float3 *p_ray,
    __global float *p_isec_dist,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_last_whichobject,
    __global const uint *p_last_which_subobject,
    __global const uint *p_inside,
    __global const float4 *vector_data,
    __global const int *integer_data)
{
    const int gid = get_global_id(0);
    
    p_whichobject += gid;
    p_which_subobject += gid;
    p_isec_dist += gid;
    
    const float3 ray = p_ray[gid];
    const float3 pos = p_pos[gid];
    
    const uint i = {{ i+1 }};
    float isec_dist = *p_isec_dist;
    uint subobject = p_last_which_subobject[gid];
    
    const uint inside_current = p_inside[gid] == i,
               origin_self = p_last_whichobject[gid] == i;
    
    float new_isec_dist = 0;
    
    // call tracer
    ### set obj = objects[i]
    ### import obj.tracer.template_file_name() as t
    
    ### if obj.tracer.convex
    if (!origin_self || inside_current) {
    ### endif
    
    ### set params = "pos, ray, isec_dist, &new_isec_dist, &subobject, inside_current, origin_self"
    ### if obj.tracer.has_data()
        __global const float4 *obj_vec_data = vector_data + {{ obj.vector_data_offset }};
        __global const int *obj_int_data = integer_data + {{ obj.integer_data_offset }};
        ### set params = "obj_vec_data, obj_int_data, " + params
    ### endif
    {{ t.tracer_call(obj.tracer, params) }}
            
    if (new_isec_dist > 0 && new_isec_dist < isec_dist)
    {
        *p_isec_dist = new_isec_dist;
        *p_whichobject = i;
        *p_which_subobject = subobject;
    }
    
    ### if obj.tracer.convex
    }
    ### endif
    
}

__kernel void shadow_trace_object_{{ i }}(
    __global const float3 *p_pos,
    __global const float3 *p_normal,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_inside,
    __global float *p_shadow_mask,
    __global const float4 *vector_data,
    __global const int *integer_data,
    constant float4 *p_dest_point)
{
    const int gid = get_global_id(0);
    if (p_shadow_mask[gid] == 0.0) return;
    
    const float3 dest = p_dest_point[0].xyz;
    const float3 pos = p_pos[gid];
    float3 ray = dest - pos;
    
    // last normal check
    if ( dot(p_normal[gid], ray) < 0.0 ) {
        p_shadow_mask[gid] = 0.0;
        return;
    }
    
    const float isec_dist = length(ray);
    ray = ray / isec_dist;
    
    uint subobject = p_which_subobject[gid];
    
    const uint i = {{ i+1 }};
    const uint inside_current = p_inside[gid] == i,
               origin_self = p_whichobject[gid] == i;
    
    float new_isec_dist = 0;
    
    // call tracer
    ### set obj = objects[i]
    ### import obj.tracer.template_file_name() as t
    
    ### if obj.tracer.convex
    if (!origin_self || inside_current) {
    ### endif
    
    ### set params = "pos, ray, isec_dist, &new_isec_dist, &subobject, inside_current, origin_self"
    ### if obj.tracer.has_data()
        __global const float4 *obj_vec_data = vector_data + {{ obj.vector_data_offset }};
        __global const int *obj_int_data = integer_data + {{ obj.integer_data_offset }};
        ### set params = "obj_vec_data, obj_int_data, " + params
    ### endif
    {{ t.tracer_call(obj.tracer, params) }}
            
    if (new_isec_dist > 0 && new_isec_dist < isec_dist)
    {
        p_shadow_mask[gid] = 0.0;
    }
    
    ### if obj.tracer.convex
    }
    ### endif
}

### endfor

__kernel void advance_and_compute_normal(
    __global float3 *p_pos,
    __global const float3 *p_ray,
    __global float3 *p_normal,
    __global const float *p_isec_dist,
    __global const uint *p_whichobject,
    __global const uint *p_which_subobject,
    __global const uint *p_inside,
    __global const float4 *vector_data,
    __global const int *integer_data)
{
    const int gid = get_global_id(0);
    
    p_whichobject += gid;
    p_which_subobject += gid;
    p_normal += gid;
    p_pos += gid;
    p_isec_dist += gid;
    
    const float3 ray = p_ray[gid];
    const uint inside = p_inside[gid];
    float3 pos = *p_pos;
    
    // advance pos along ray by isec_dist
    pos += (*p_isec_dist) * ray;
    *p_pos = pos;
    
    const uint whichobject = *p_whichobject;
    const uint subobject = *p_which_subobject;
        
    __global const float4 *obj_vec_data;
    __global const int *obj_int_data;
    
    ### for i in range(n_objects)
        
        {% if i %}else {% endif %}if (whichobject == {{ i + 1 }})
        {
            // call normal
            ### set obj = objects[i]
            ### import obj.tracer.template_file_name() as t
            
            ### set params = "pos, subobject, p_normal"
            ### if obj.tracer.has_data()
                obj_vec_data = vector_data + {{ obj.vector_data_offset }};
                obj_int_data = integer_data + {{ obj.integer_data_offset }};
                ### set params = "obj_vec_data, obj_int_data, " + params
            ### endif
            {{ t.normal_call(obj.tracer, params) }}
            
            ### if obj.tracer.auto_flip_normal
                if (dot(*p_normal, ray) > 0) *p_normal = -*p_normal;
            ### else
                if (inside == {{ i + 1 }}) *p_normal = -*p_normal;
            ### endif
        }
        
    ### endfor
}

