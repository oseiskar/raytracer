
### from 'object.cl' import tracer_params

#define DATA_POINTER_BUFFER_OFFSET {{renderer.object_data_pointer_buffer_offset}}

#define DATA_float3 0
#define DATA_int 1
#define DATA_PARAM_float3 2
#define DATA_PARAM_int 3
#define DATA_PARAM_float 4
#define DATA_N_TYPES 5

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
    
    constant int *data_offsets = param_int_data + DATA_N_TYPES*(whichobject-1) + DATA_POINTER_BUFFER_OFFSET;
    
    ### for i in range(n_objects)
    
        {% if i %}else {% endif %}if (whichobject == {{ i + 1 }})
        {
            // call normal
            ### set obj = objects[i]
            
            {{ obj.tracer.normal_function_name }}(
                pos, subobject, p_normal
                {{ tracer_params(obj.tracer) }}
            );
            
            ### if obj.tracer.auto_flip_normal
                if (dot(*p_normal, ray) > 0) *p_normal = -*p_normal;
            ### else
                if (inside == {{ i + 1 }}) *p_normal = -*p_normal;
            ### endif
        }
        
    ### endfor
}

