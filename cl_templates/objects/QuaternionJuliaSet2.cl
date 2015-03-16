### extends 'object.cl'

### from 'objects/bounding_volumes.cl' import sphere_bounding_volume

### macro tracer_function(obj)
    ### call tracer_function_base(obj)
    
		if (origin_self) return; // ------------- remember to remove this
		float trace_begin, trace_end;
        
        {{ sphere_bounding_volume(obj.center, obj.bndR*obj.scale, 'trace_begin', 'trace_end') }}
        
        int i=0;
		const int MAX_ITER = {{ obj.max_itr }};
		const float TARGET_EPS = {{ obj.precision }};
		const float MAX_STEP = {{ obj.max_step }};
		const float ESCAPE_RADIUS = 2.0f;
		
		float step, dist;
		const float4 c = {{ obj.c_as_float4 }};
		dist = trace_begin;
		float3 pos = trace_begin * ray + origin - {{vec3(obj.center)}};
		
		for( i=0; i < MAX_ITER; i++ )
		{
			float4 q = (float4)(pos,0), q1, qd = (float4)(1,0,0,0);
			
			float gr = ray.x, gi = ray.y, gj = ray.z, gk = 0,
				  gr1, gi1, gj1, gk1;
			
			for (int iii=0; iii<{{ obj.julia_itr }}; ++iii)
			{
				if (origin_self)
				{
					// TODO: use quaternion derivative, if possible
				
					// Derivative chain rule
					gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
					gi1 = 2*(gr*q.y + q.x*gi);
					gj1 = 2*(gr*q.z + q.x*gj);
					gk1 = 2*(gr*q.w + q.x*gk);
					
					gr = gr1;
					gi = gi1;
					gj = gj1;
					gk = gk1;
				}
				
				qd = 2*quaternion_mult(q,qd);
				q = quaternion_square(q) + c;
				
			}
			
			float ql2 = dot(q,q);
			float ds = 2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk);
			
			// The magic distance estimate formula, see the 1989 article:
			// Hart, Sandin, Kauffman, "Ray Tracing Deterministic 3-D Fractals"
			float l = length(q);
			step = 0.5 * l * log(l) / length(qd);
			step = min(step, MAX_STEP);
			
			if (step < TARGET_EPS)
			{
				if (!origin_self || ds < 0)
				{ 
					*p_new_isec_dist = dist + step;
					return;
				}
				else
				{
					step = TARGET_EPS;
				}
			}
			
			dist += step;
			
			if (dist > trace_end) return;
			pos += step * ray;
		}
        
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
        
		float4 q = (float4)(pos - {{ vec3(obj.center) }},0);
		const float4 c = {{ obj.c_as_float4 }}, q1;
		
		float3
			gr = (float3)(1,0,0),
			gi = (float3)(0,1,0),
			gj = (float3)(0,0,1),
			gk = (float3)(0,0,0),
			gr1, gi1, gj1, gk1;
		
		float qr1;
		
		for (int iii=0; iii<{{ obj.julia_itr }}; ++iii)
		{
			// Derivative chain rule
			gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
			gi1 = 2*(gr*q.y + q.x*gi);
			gj1 = 2*(gr*q.z + q.x*gj);
			gk1 = 2*(gr*q.w + q.x*gk);
			
			gr = gr1;
			gi = gi1;
			gj = gj1;
			gk = gk1;
			
			// Quaternion operation z -> z^2 + c
			q = quaternion_square(q) + c;
		}
		float3 grad = fast_normalize(2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk));
		
		if (all(isfinite(grad))) *p_normal = grad;
		else *p_normal = (float3)(1,0,0);
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}});
### endmacro
