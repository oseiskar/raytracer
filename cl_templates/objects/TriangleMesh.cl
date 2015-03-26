
 
// adapted from http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

### extends 'object.cl'

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'uint n_triangles')
    
        const float EPSILON = 0.00001;
        uint old_subobject = *p_subobject, subobject;
        float isec_dist = old_isec_dist;
        
        for (uint j=0; j<n_triangles; ++j) {
            
            if (old_subobject == j && origin_self) continue;
            
            const float3 v1 = vector_data[integer_data[3*j]].xyz;
            const float3 v2 = vector_data[integer_data[3*j+1]].xyz;
            const float3 v3 = vector_data[integer_data[3*j+2]].xyz;
            
            const float3 e1 = v2 - v1;
            const float3 e2 = v3 - v1;
            
            //Begin calculating determinant - also used to calculate u parameter
            const float3 P = cross(ray, e2);
            //if determinant is near zero, ray lies in plane of triangle
            const float det = dot(e1, P);
            
            if (det > -EPSILON && det < EPSILON) continue;
            
            const float inv_det = 1.f / det;

            //calculate distance from V1 to ray origin
            const float3 T = origin - v1;

            //Calculate u parameter and test bound
            const float u = dot(T, P) * inv_det;
            //The intersection lies continue of the triangle
            if(u < 0.f || u > 1.f) continue;

            //Prepare to test v parameter
            const float3 Q = cross(T, e1);

            //Calculate V parameter and test bound
            const float v = dot(ray, Q) * inv_det;
            //The intersection lies outside of the triangle
            if(v < 0.f || u + v  > 1.f) continue;

            const float t = dot(e2, Q) * inv_det;

            if(t > 0.0 && t < isec_dist) { //ray intersection
                isec_dist = t;
                subobject = j;
            }
        }
        
        if (isec_dist < old_isec_dist) {
            *p_new_isec_dist = isec_dist;
            *p_subobject = subobject;
        }
        
    
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj)
    
        const float3 v1 = vector_data[integer_data[3*subobject]].xyz;
        const float3 v2 = vector_data[integer_data[3*subobject+1]].xyz;
        const float3 v3 = vector_data[integer_data[3*subobject+2]].xyz;
        
        *p_normal = fast_normalize(cross(v2-v1, v3-v1));
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{obj.n_faces}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}});
### endmacro
