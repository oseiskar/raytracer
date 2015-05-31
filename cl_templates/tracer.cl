
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
    __global const int *pixel,
    __global const float3 *p_pos,
    __global const float3 *p_ray,
    __global float *p_isec_dist,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_last_whichobject,
    __global const uint *p_last_which_subobject,
    __global const uint *p_inside,
    TRACER_DATA const float4 *vector_data,
    TRACER_DATA const int *integer_data,
    constant const float4 *param_float3_data,
    constant const int *param_int_data,
    constant const float *param_float_data,
    int object_id)
{
    const int thread_idx = get_global_id(0);
    const int ray_idx = pixel[thread_idx];
    
    const float old_isec_dist = p_isec_dist[thread_idx];
    float isec_dist = old_isec_dist;
    const uint old_subobject =  p_last_which_subobject[ray_idx];
    uint subobject, whichobject;
    
    const uint inside_current = p_inside[ray_idx] == object_id,
               origin_self = p_last_whichobject[ray_idx] == object_id;
    
    constant const int *data_offsets = param_int_data + DATA_N_TYPES*(object_id-1) + DATA_POINTER_BUFFER_OFFSET;
    
    // call tracer
    
    ### if obj.convex
    if (!origin_self || inside_current) {
    ### endif

        float new_isec_dist = 0;
        uint cur_subobject = old_subobject;
    
        {{ obj.tracer_function_name }}(
            p_pos[ray_idx], p_ray[ray_idx], isec_dist, &new_isec_dist, &cur_subobject,
            inside_current, origin_self
            {{ tracer_params(obj) }}
        );
                
        if (new_isec_dist > 0 && new_isec_dist < isec_dist)
        {
            p_isec_dist[thread_idx] = new_isec_dist;
            p_which_subobject[ray_idx] = cur_subobject;
            p_whichobject[ray_idx] = object_id;
        }
    
    ### if obj.convex
    }
    ### endif
}


### endmacro

### macro shadow_kernel(obj)

__kernel void {{ obj.shadow_kernel_name }}(
    __global const int *pixel,
    __global const float3 *p_pos,
    __global const float3 *p_normal,
    __global uint *p_whichobject,
    __global uint *p_which_subobject,
    __global const uint *p_inside,
    __global float *p_shadow_mask,
    TRACER_DATA const float4 *vector_data,
    TRACER_DATA const int *integer_data,
    constant const float4 *param_float3_data,
    constant const int *param_int_data,
    constant const float *param_float_data,
    constant float4 *p_dest_point,
    int light_id,
    int offset, int count)
{
    const int thread_idx = get_global_id(0);
    const int ray_idx = pixel[thread_idx];
    
    if (p_shadow_mask[thread_idx] == 0.0) return;
    
    const int object_index = get_global_id(1);
    //if (object_index >= count) return;
    const int object_id = object_index + offset;
    if (object_id == light_id) return;
    
    const float3 dest = p_dest_point[0].xyz;
    const float3 pos = p_pos[ray_idx];
    float3 ray = dest - pos;
    
    // last normal check
    if ( dot(p_normal[thread_idx], ray) < 0.0 ) {
        p_shadow_mask[thread_idx] = 0.0;
        return;
    }
    
    const float isec_dist = length(ray);
    ray = ray / isec_dist;
    
    uint subobject = p_which_subobject[ray_idx];
    
    const uint inside_current = p_inside[ray_idx] == object_id,
               origin_self = p_whichobject[ray_idx] == object_id;
    
    float new_isec_dist = 0;
    
    constant const int *data_offsets = param_int_data + DATA_N_TYPES*(object_id-1) + DATA_POINTER_BUFFER_OFFSET;
    
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
        p_shadow_mask[thread_idx] = 0.0;
    }
    
    ### if obj.convex
    }
    ### endif
}

### endmacro

### macro normal_kernel(obj)

__kernel void {{ obj.normal_kernel_name }}(
    __global const int *pixel,
    __global float3 *p_pos,
    __global const float3 *p_ray,
    __global float3 *p_normal,
    __global const float *p_isec_dist,
    __global const uint *p_whichobject,
    __global const uint *p_which_subobject,
    __global const uint *p_inside,
    TRACER_DATA const float4 *vector_data,
    TRACER_DATA const int *integer_data,
    constant float4 *param_float3_data,
    constant int *param_int_data,
    constant float *param_float_data,
    int offset, int count)
{
    const int thread_idx = get_global_id(0);
    const int ray_idx = pixel[thread_idx];
    
    p_whichobject += ray_idx;
    const uint whichobject = *p_whichobject;
    
    if (whichobject >= offset && whichobject < offset+count) {
    
        p_which_subobject += ray_idx;
        p_normal += thread_idx;
        p_pos += ray_idx;
        p_isec_dist += thread_idx;
        
        const float3 ray = p_ray[ray_idx];
        const uint inside = p_inside[ray_idx];
        float3 pos = *p_pos;
        
        // advance pos along ray by isec_dist
        pos += (*p_isec_dist) * ray;
        *p_pos = pos;
        const uint subobject = *p_which_subobject;
        
        constant const int *data_offsets = param_int_data + DATA_N_TYPES*(whichobject-1) + DATA_POINTER_BUFFER_OFFSET;
        
        {{ obj.normal_function_name }}(
            pos, subobject, p_normal
            {{ tracer_params(obj) }}
        );
        
        {#/* TODO: move this to triangle mesh */#}
        ### if obj.auto_flip_normal
            if (dot(*p_normal, ray) > 0) *p_normal = -*p_normal;
        ### else
            if (inside == whichobject) *p_normal = -*p_normal;
        ### endif
    }
}

### endmacro
