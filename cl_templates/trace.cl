uint trace_core(
    float3 ray,
    const float3 last_normal,
    const uint last_whichobject,
    const uint inside,
    __private float3 *p_pos,
    __private uint *p_subobject,
    __private float *p_isec_dist)
{

    float3 pos = *p_pos;
    float old_isec_dist = *p_isec_dist;
    uint whichobject = 0, subobject, cur_subobject, i;
    
    float new_isec_dist = 0;

    ### for i in range(objects.length)
        
        new_isec_dist = 0;
        i = {{ i+1 }};
        
        // call tracer
        {{ objects.tracers[i].make_tracer_call("pos, ray, last_normal, old_isec_dist, &new_isec_dist, &cur_subobject, inside == i, last_whichobject == i") }}
                
        if (//last_whichobject != i && // cull self
            new_isec_dist > 0 &&
            new_isec_dist < old_isec_dist)
        {
            old_isec_dist = new_isec_dist;
            whichobject = i;
            subobject = cur_subobject;
        }
        
    
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
    __global const uint *p_inside,
    __global float *p_shadow_mask,
    // a random unit vector and color mask
    constant float4 *p_dest_point,
    uint light_id)
{
    const int gid = get_global_id(0);
    
    const float3 dest = p_dest_point[0].xyz;
    float3 pos = p_pos[gid];
    float shadow_dist = length(dest - pos);
    const float3 ray = fast_normalize(dest - pos);
    const float3 last_normal = p_normal[gid];
    
    if ( dot(last_normal, ray) < 0.0 ) {
        p_shadow_mask[gid] = 0.0;
        return;
    }
    
    float isec_dist = shadow_dist;
    uint subobject;
    
    uint whichobject = trace_core(
        ray,
        last_normal,
        p_whichobject[gid],
        p_inside[gid],
        &pos,&subobject,&isec_dist);
    
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
    __global const uint *p_inside)
{
    const int gid = get_global_id(0);
    
    p_whichobject += gid;
    p_normal += gid;
    p_isec_dist += gid;
    p_pos += gid;
    
    const float3 ray = p_ray[gid];
    const uint inside = p_inside[gid];
    
    float3 pos = *p_pos;
    float isec_dist = *p_isec_dist;
    uint subobject;
    
    uint whichobject = trace_core(
        ray,
        *p_normal,
        *p_whichobject,
        inside,
        &pos,&subobject,&isec_dist);
    
    ### for i in range(objects.length)
        
        {% if i %}else {% endif %}if (whichobject == {{ i + 1 }})
        {
            // call normal
            {{ objects.tracers[i].make_normal_call("pos, subobject, p_normal") }}
            if (inside == {{ i + 1 }}) *p_normal = -*p_normal;
        }
        
    ### endfor

    *p_isec_dist = isec_dist;
    *p_whichobject = whichobject;
    *p_pos = pos;
}
