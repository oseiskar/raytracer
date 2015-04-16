
{% include 'static.cl' %}

{% include 'shade.cl' %}

// Declarations
### for k in functions.declarations
{{ k }}
### endfor

// Definitions
### for k in functions.definitions
{{ k }}
### endfor


### include 'trace.cl'

// Tracer kernels
### for k in functions.kernels
{{ k }}
### endfor
