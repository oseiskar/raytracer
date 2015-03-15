
{% include 'utils.cl' %}

{% include 'shade.cl' %}

// Declarations
### for k in kernels.declarations
{{ k }}
### endfor


// Definitions
### for k in kernels.functions
{{ k }}
### endfor

### include 'trace.cl'
