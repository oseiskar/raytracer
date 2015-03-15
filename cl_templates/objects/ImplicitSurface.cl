### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
        {{ obj.tracer_code }}
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        {{ obj.normal_code }}
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}});
### endmacro
