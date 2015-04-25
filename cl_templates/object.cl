
### macro normal_function_base(obj, extra_params)
{#/*
    The normal functions are supposed to compute the exterior normal
    at the given point
*/#}
void {{ obj.normal_function_name }}(
        const float3 pos,            {# a point on the surface of the object #}
        const uint subobject,        {# subobject number (computed by the tracer) #}
        __global float3 *p_normal    {# [out] the computed normal #}
        {{ obj.parameter_declaration_string() }}) {
{{ caller() }}
}
### endmacro

### macro tracer_function_base(obj)
{#/*
    The tracer functions are supposed compute the distance to nearest valid
    intersection of the given ray and the object that the tracer function
    represents
*/#}
void {{ obj.tracer_function_name }}(
        const float3 origin,              {# ray origin #}
        const float3 ray,                 {# ray direction #}
        const float old_isec_dist,        {# upper bound for isec. distance #}
        __private float *p_new_isec_dist, {# [out] computed isec. distance #}
        __private uint *p_subobject,      {# [out] (optional) subobject number (e.g., which face of a cube),
                                             passed to the normal computation function #}
        bool inside,                      {# is the ray travelling inside the object #}
        bool origin_self                  {# self-intersection? #}
        {{ obj.parameter_declaration_string() }}) {
{{ caller() }}
}
### endmacro

### macro tracer_component_function_base(obj)
void {{ obj.tracer_function_name }}(
        const float3 origin,            {# ray origin #}
        const float3 ray,               {# ray direction  #}
        __private float *p_isec_begin,  {# [out] computed isec. interval begin distance #}
        __private float *p_isec_end,    {# [out] computed isec. interval end distance #}
        __private uint *p_subobject,    {# [out] (optional) subobject number (e.g., which face of a cube),
                                            passed to the normal computation function #}
        bool inside                     {# is the ray travelling inside the object #}
        {{ obj.parameter_declaration_string() }}) {
{{ caller() }}
}
### endmacro

### macro tracer_params(obj)

    ### if obj.has_data()
        ,
        vector_data + data_offsets[DATA_float3],
        integer_data + data_offsets[DATA_int]
    ### endif

    ### set names = obj.parameter_declarations()
    ### set n_params = names|length
    ### for i in range(n_params)
        ### set cl_type = obj.parameter_types()[i]
        , param_{{ cl_type }}_data[data_offsets[DATA_PARAM_{{cl_type}}] + {{ obj.local_param_offsets[i] }}]{% if cl_type == 'float3' %}.xyz{% endif %} // {{ names[i] }}
    ### endfor

### endmacro

### macro tracer_kernel(obj)

__kernel void {{ obj.tracer_kernel_name }}(
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
    constant const float4 *param_float3_data,
    constant const int *param_int_data,
    constant const float *param_float_data,
    int offset, int count)
{
    const int gid = get_global_id(0);
    const float old_isec_dist = p_isec_dist[gid];
    float isec_dist = old_isec_dist;
    const uint old_subobject =  p_last_which_subobject[gid];
    uint subobject, whichobject;
    
    for (uint i = offset; i < offset + count; i++)
    {
        const uint inside_current = p_inside[gid] == i,
                   origin_self = p_last_whichobject[gid] == i;
        
        constant int *data_offsets = param_int_data + DATA_N_TYPES*(i-1) + DATA_POINTER_BUFFER_OFFSET;
        
        // call tracer
        
        ### if obj.convex
        if (!origin_self || inside_current) {
        ### endif
    
            float new_isec_dist = 0;
            uint cur_subobject = old_subobject;
        
            {{ obj.tracer_function_name }}(
                p_pos[gid], p_ray[gid], isec_dist, &new_isec_dist, &cur_subobject,
                inside_current, origin_self
                {{ tracer_params(obj) }}
            );
                    
            if (new_isec_dist > 0 && new_isec_dist < isec_dist)
            {
                isec_dist = new_isec_dist;
                subobject = cur_subobject;
                whichobject = i;
            }
        
        ### if obj.convex
        }
        ### endif
    
    }
    
    if (isec_dist < old_isec_dist)
    {
        p_isec_dist[gid] = isec_dist;
        p_whichobject[gid] = whichobject;
        p_which_subobject[gid] = subobject;
    }
}


### endmacro

### macro shadow_kernel(obj)

__kernel void {{ obj.shadow_kernel_name }}(
    __global const float3 *p_pos,
    __global const float3 *p_normal,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_inside,
    __global float *p_shadow_mask,
    __global const float4 *vector_data,
    __global const int *integer_data,
    constant const float4 *param_float3_data,
    constant const int *param_int_data,
    constant const float *param_float_data,
    constant float4 *p_dest_point,
    int object_id)
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
    
    const uint i = object_id;
    const uint inside_current = p_inside[gid] == i,
               origin_self = p_whichobject[gid] == i;
    
    float new_isec_dist = 0;
    
    constant int *data_offsets = param_int_data + DATA_N_TYPES*(i-1) + DATA_POINTER_BUFFER_OFFSET;
    
    // call tracer
    
    ### if obj.convex
    if (!origin_self || inside_current) {
    ### endif
    
        {{ obj.tracer_function_name }}(
            pos, ray, isec_dist, &new_isec_dist, &subobject, inside_current, origin_self
            {{ tracer_params(obj) }}
        );
    
    if (new_isec_dist > 0 && new_isec_dist < isec_dist)
    {
        p_shadow_mask[gid] = 0.0;
    }
    
    ### if obj.convex
    }
    ### endif
}

### endmacro
