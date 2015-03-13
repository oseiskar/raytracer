
### include 'utils.cl'

### include 'shade.cl'

### for k in kernels.declarations
    {{ k }}
### endfor

### for k in kernels.functions
    {{ k }}
### endfor

### include 'trace.cl'
