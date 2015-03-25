uint trace_core(
    float3 ray,
    const uint last_whichobject,
    const uint inside,
    __private float3 *p_pos,
    __private uint *p_subobject,
    __private float *p_isec_dist,
    __global const float4 *vector_data)
{

    float3 pos = *p_pos;
    float old_isec_dist = *p_isec_dist;
    uint whichobject = 0, subobject = *p_subobject, cur_subobject, i, origin_self, inside_current;
    
    float new_isec_dist = 0;
    __global const float3 *obj_vec_data;

    ### for i in range(n_objects)
        
        new_isec_dist = 0;
        i = {{ i+1 }};
        
        // call tracer
        ### set obj = objects[i]
        ### import obj.tracer.template_file_name() as t
        
        origin_self = last_whichobject == i;
        inside_current = inside == i;
        
        ### if obj.tracer.convex
        if (!origin_self || inside_current) {
        ### endif
        
        if (origin_self) cur_subobject = *p_subobject;
        
        ### set params = "pos, ray, old_isec_dist, &new_isec_dist, &cur_subobject, inside_current, origin_self"
        ### if obj.tracer.has_vector_data()
            obj_vec_data = vector_data + {{ obj.vector_data_offset }};
            ### set params = "obj_vec_data, " + params
        ### endif
        {{ t.tracer_call(obj.tracer, params) }}
                
        if (//last_whichobject != i && // cull self
            new_isec_dist > 0 &&
            new_isec_dist < old_isec_dist)
        {
            old_isec_dist = new_isec_dist;
            whichobject = i;
            subobject = cur_subobject;
        }
        
        ### if obj.tracer.convex
        }
        ### endif
        
    
    ### endfor
    
    pos += old_isec_dist * ray; // saxpy
        
    *p_pos = pos;
    *p_subobject = subobject;
    *p_isec_dist = old_isec_dist;
    
    return whichobject;
}
    
__kernel void shadow_trace(
    __global const float3 *p_pos,
    __global const float3 *p_normal,
    __global const uint *p_whichobject,
    __global const uint *p_which_subobject,
    __global const uint *p_inside,
    __global float *p_shadow_mask,
    __global const float4 *vector_data,
    // a random unit vector and color mask
    constant float4 *p_dest_point,
    uint light_id)
{
    const int gid = get_global_id(0);
    
    const float3 dest = p_dest_point[0].xyz;
    float3 pos = p_pos[gid];
    float shadow_dist = length(dest - pos);
    const float3 ray = fast_normalize(dest - pos);
    
    // last normal check
    if ( dot(p_normal[gid], ray) < 0.0 ) {
        p_shadow_mask[gid] = 0.0;
        return;
    }
    
    float isec_dist = shadow_dist;
    uint subobject = p_which_subobject[gid];
    
    uint whichobject = trace_core(
        ray,
        p_whichobject[gid],
        p_inside[gid],
        &pos,&subobject,&isec_dist,
        vector_data);
    
    // no light self-intersection (light objects must be convex)
    if (whichobject == 0 || whichobject == light_id) {
        // no shadow
        p_shadow_mask[gid] = 1.0;
    }
    else {
        // shadow
        p_shadow_mask[gid] = 0.0;
    }
}

__kernel void trace(
    __global float3 *p_pos,
    __global const float3 *p_ray,
    __global float3 *p_normal,
    __global float *p_isec_dist,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_inside,
    __global const float4 *vector_data)
{
    const int gid = get_global_id(0);
    
    p_whichobject += gid;
    p_which_subobject += gid;
    p_normal += gid;
    p_isec_dist += gid;
    p_pos += gid;
    
    const float3 ray = p_ray[gid];
    const uint inside = p_inside[gid];
    
    float3 pos = *p_pos;
    float isec_dist = *p_isec_dist;
    uint subobject = *p_which_subobject;
    
    uint whichobject = trace_core(
        ray,
        *p_whichobject,
        inside,
        &pos,&subobject,&isec_dist,
        vector_data);
        
    __global const float3 *obj_vec_data;
    
    ### for i in range(n_objects)
        
        {% if i %}else {% endif %}if (whichobject == {{ i + 1 }})
        {
            // call normal
            ### set obj = objects[i]
            ### import obj.tracer.template_file_name() as t
            
            ### set params = "pos, subobject, p_normal"
            ### if obj.tracer.has_vector_data()
                obj_vec_data = vector_data + {{ obj.vector_data_offset }};
                ### set params = "obj_vec_data, " + params
            ### endif
            {{ t.normal_call(obj.tracer, params) }}
            
            ### if obj.tracer.auto_flip_normal
                if (dot(*p_normal, ray) > 0) *p_normal = -*p_normal;
            ### else
                if (inside == {{ i + 1 }}) *p_normal = -*p_normal;
            ### endif
        }
        
    ### endfor

    *p_isec_dist = isec_dist;
    *p_whichobject = whichobject;
    *p_which_subobject = subobject;
    *p_pos = pos;
}
