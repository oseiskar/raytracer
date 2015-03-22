### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'const float3 base')
    
		if (origin_self && !inside) return;
		
		float3 rel = origin - base;
		float ibegin  = 0.0, iend = old_isec_dist;
		float cur_ibegin, cur_iend;
		uint subobj, cur_subobj;
		
		### set subobj_offset = 0
        ### for c in obj.components
            
			cur_subobj = 0;
			cur_ibegin = ibegin;
			cur_iend = iend;
            
            ### import c.template_file_name() as t
            {{ t.tracer_call(c, obj.component_tracer_call_params(c)) }}
			
			if (cur_ibegin > ibegin) {
				ibegin = cur_ibegin;
				if (!inside) subobj = cur_subobj + {{ subobj_offset }};
			}
			if (cur_iend < iend) {
				iend = cur_iend;
				if (inside) subobj = cur_subobj + {{ subobj_offset }};
			}
			if (ibegin > iend || ibegin > old_isec_dist) return;
			
			### set subobj_offset = subobj_offset + c.n_subobjects
		
        ### endfor
        
        
		if (inside) *p_new_isec_dist = iend;
		else *p_new_isec_dist = ibegin;
		*p_subobject = subobj;
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'const float3 base')
    
		const float3 p = pos - base;
		
		### set subobj_offset = 0
        ### for c in obj.components
        
            {% if subobj_offset > 0 %}else {% endif %}if ({% if subobj_offset > 0 %}subobject >= {{ subobj_offset }} && {% endif %}subobject < {{ subobj_offset + c.n_subobjects }})
            
                ### import c.template_file_name() as t
                {{ t.normal_call(c, obj.component_normal_call_params(c, subobj_offset)) }}
                
			### set subobj_offset = subobj_offset + c.n_subobjects
        
        ### endfor
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{vec3(obj.origin)}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{vec3(obj.origin)}});
### endmacro
