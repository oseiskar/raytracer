
### from 'helpers.cl' import vec3

### macro normal_function_base(obj, extra_params)
{#/*
    The normal functions are supposed to compute the exterior normal
    at the given point
*/#}
void {{ obj.normal_function_name }}(
    ### if obj.has_data()
        __global const float4 *vector_data,   {# vector data for this object #}
        __global const int *integer_data,     {# integer data for this object #}
    ### endif
        const float3 pos,            {# a point on the surface of the object #}
        const uint subobject,        {# subobject number (computed by the tracer) #}
        __global float3 *p_normal    {# [out] the computed normal #}
        {%- if extra_params -%},{%- endif %}
        {{ extra_params }}) {
{{ caller() }}
}
### endmacro

### macro tracer_function_base(obj, extra_params)
{#/*
    The tracer functions are supposed compute the distance to nearest valid
    intersection of the given ray and the object that the tracer function
    represents
*/#}
void {{ obj.tracer_function_name }}(
    ### if obj.has_data()
        __global const float4 *vector_data,   {# vector data for this object #}
        __global const int *integer_data,     {# integer data for this object #}
    ### endif
        const float3 origin,              {# ray origin #}
        const float3 ray,                 {# ray direction #}
        const float old_isec_dist,        {# upper bound for isec. distance #}
        __private float *p_new_isec_dist, {# [out] computed isec. distance #}
        __private uint *p_subobject,      {# [out] (optional) subobject number (e.g., which face of a cube),
                                             passed to the normal computation function #}
        bool inside,                      {# is the ray travelling inside the object #}
        bool origin_self                  {# self-intersection? #}
        {%- if extra_params -%},{%- endif %}
        {{ extra_params }}) {
{{ caller() }}
}
### endmacro

### macro tracer_component_function_base(obj, extra_params)
void {{ obj.tracer_function_name }}(
        const float3 origin,            {# ray origin #}
        const float3 ray,               {# ray direction  #}
        __private float *p_isec_begin,  {# [out] computed isec. interval begin distance #}
        __private float *p_isec_end,    {# [out] computed isec. interval end distance #}
        __private uint *p_subobject,    {# [out] (optional) subobject number (e.g., which face of a cube),
                                            passed to the normal computation function #}
        bool inside                     {# is the ray travelling inside the object #}
        {%- if extra_params -%},{%- endif %}   
        {{ extra_params }}) {
{{ caller() }}
}
### endmacro
