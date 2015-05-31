
{% include 'static.cl' %}

{% include 'shader.cl' %}

#define TRACER_DATA __constant

// Declarations
### for k in functions.declarations
{{ k }}
### endfor

// Definitions
### for k in functions.definitions
{{ k }}
### endfor

#define DATA_POINTER_BUFFER_OFFSET {{renderer.object_data_pointer_buffer_offset}}

#define DATA_float3 0
#define DATA_int 1
#define DATA_PARAM_float3 2
#define DATA_PARAM_int 3
#define DATA_PARAM_float 4
#define DATA_N_TYPES 5

// Tracer kernels
### for k in functions.kernels
{{ k }}
### endfor
