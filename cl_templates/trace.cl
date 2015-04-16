
#define DATA_float3 0
#define DATA_int 1
#define DATA_PARAM_float3 2
#define DATA_PARAM_int 3
#define DATA_PARAM_float 4
#define DATA_N_TYPES 5

### macro tracer_params(obj)
    ### set names = obj.tracer.parameter_declarations()
    ### set n_params = names|length
    ### for i in range(n_params)
        ### set cl_type = obj.tracer.parameter_types()[i]
        , param_{{ cl_type }}_data[data_offsets[DATA_PARAM_{{cl_type}}] + {{ obj.local_param_offsets[i] }}]{% if cl_type == 'float3' %}.xyz{% endif %} // {{ names[i] }}
    ### endfor

### endmacro

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
    __global const int *integer_data,
    constant const float4 *param_vector_data,
    constant const int *param_integer_data,
    constant const float *param_float_data)
{
    const int gid = get_global_id(0);
    
    const uint i = {{ i+1 }};
    uint subobject = p_last_which_subobject[gid];
    
    const uint inside_current = p_inside[gid] == i,
               origin_self = p_last_whichobject[gid] == i;
    
    float isec_dist = p_isec_dist[gid];
    float new_isec_dist = 0;
    
    // call tracer
    ### set obj = objects[i]
    ### import obj.tracer.template_file_name() as t
    
    ### if obj.tracer.convex
    if (!origin_self || inside_current) {
    ### endif
    
    ### set params = "p_pos[gid], p_ray[gid], isec_dist, &new_isec_dist, &subobject, inside_current, origin_self"
    ### if obj.tracer.has_data()
        __global const float4 *obj_vec_data = vector_data + {{ obj.vector_data_offset }};
        __global const int *obj_int_data = integer_data + {{ obj.integer_data_offset }};
        ### set params = "obj_vec_data, obj_int_data, " + params
    ### endif
    {{ t.tracer_call(obj.tracer, params) }}
            
    if (new_isec_dist > 0 && new_isec_dist < isec_dist)
    {
        p_isec_dist[gid] = new_isec_dist;
        p_whichobject[gid] = i;
        p_which_subobject[gid] = subobject;
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
    constant const float4 *param_vector_data,
    constant const int *param_integer_data,
    constant const float *param_float_data,
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
    __global const int *integer_data,
    constant float4 *param_float3_data,
    constant int *param_int_data,
    constant float *param_float_data)
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
    
    constant int *data_offsets = param_int_data + DATA_N_TYPES*(whichobject-1) + {{shader.object_data_pointer_buffer_offset}};
    
    ### for i in range(n_objects)
    
        {% if i %}else {% endif %}if (whichobject == {{ i + 1 }})
        {
            // call normal
            ### set obj = objects[i]
            ### import obj.tracer.template_file_name() as t
            
            {{ obj.tracer.normal_function_name }}(
            ### if obj.tracer.has_data()
                vector_data + data_offsets[DATA_float3],
                integer_data + data_offsets[DATA_int],
            ### endif
                pos, subobject, p_normal
                {{ tracer_params(obj) }}
            );
            
            ### if obj.tracer.auto_flip_normal
                if (dot(*p_normal, ray) > 0) *p_normal = -*p_normal;
            ### else
                if (inside == {{ i + 1 }}) *p_normal = -*p_normal;
            ### endif
        }
        
    ### endfor
}

