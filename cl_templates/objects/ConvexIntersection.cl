### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
		
		float3 rel = origin - base;
		float ibegin  = 0.0, iend = old_isec_dist;
		float cur_ibegin, cur_iend;
		uint subobj, cur_subobj, last_subobject = *p_subobject;
		
		### set subobj_offset = 0
        ### for c in obj.components
            
			cur_subobj = 0;
			cur_ibegin = ibegin;
			cur_iend = iend;
            
            if (origin_self && last_subobject > {{ subobj_offset }})
                cur_subobj = last_subobject - {{ subobj_offset }};
            
            {{ c.tracer_function_name }}(
                {{ obj.component_tracer_call_params(c) }}
                {{ c.parameter_value_string() }} // TODO
            );
			
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
    ### call normal_function_base(obj)
    
		const float3 p = pos - base;
		
		### set subobj_offset = 0
        ### for c in obj.components
        
            {% if subobj_offset > 0 %}else {% endif %}if ({% if subobj_offset > 0 %}subobject >= {{ subobj_offset }} && {% endif %}subobject < {{ subobj_offset + c.n_subobjects }})
            
                {{ c.normal_function_name }}(
                    {{ obj.component_normal_call_params(c, subobj_offset) }}
                    {{ c.parameter_value_string() }} // TODO
                );
                
			### set subobj_offset = subobj_offset + c.n_subobjects
        
        ### endfor
        
    ### endcall
### endmacro
