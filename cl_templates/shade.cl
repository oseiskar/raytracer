
### if renderer.bidirectional
    #define BIDIRECTIONAL
### endif

### for item in shader.get_material_property_offsets()
    #define MAT_{{item[0]}} {{item[1]}}
### endfor

### include shader.shader_name + '.cl'
