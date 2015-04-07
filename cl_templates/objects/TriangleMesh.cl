
 
// adapted from http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

### extends 'object.cl'

### macro tracer_function_core()

    for (uint j=0; j<n_triangles; ++j) {
        
        const int face_index = {{ caller('j') }};
        
        if (old_subobject == face_index && origin_self) continue;
        
        const int v1i = integer_data[3*face_index];
        const int v2i = integer_data[3*face_index+1];
        const int v3i = integer_data[3*face_index+2];
        
        const float3 v1 = vector_data[v1i].xyz;
        const float3 v2 = vector_data[v2i].xyz;
        const float3 v3 = vector_data[v3i].xyz;
        
        const float3 e1 = v2 - v1;
        const float3 e2 = v3 - v1;
        
        //Begin calculating determinant - also used to calculate u parameter
        const float3 P = cross(ray, e2);
        //if determinant is near zero, ray lies in plane of triangle
        const float det = dot(e1, P);
        
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
            subobject = face_index;
        }
    }
    
### endmacro

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'uint n_triangles')
        
        uint old_subobject = *p_subobject, subobject;
        float isec_dist = old_isec_dist;
        
        ### call(triangle_number) tracer_function_core()
            // no face pointers in tracing just loop through all faces
            {{ triangle_number }}
        ### endcall
        
        if (isec_dist < old_isec_dist) {
            *p_new_isec_dist = isec_dist;
            *p_subobject = subobject;
        }
    ### endcall
### endmacro

### macro normal_function(obj)
    ### call normal_function_base(obj, 'uint n_vertices')
        const int v1i = integer_data[3*subobject],
                  v2i = integer_data[3*subobject+1],
                  v3i = integer_data[3*subobject+2];
        
        const float3 v1 = vector_data[v1i].xyz;
        const float3 v2 = vector_data[v2i].xyz;
        const float3 v3 = vector_data[v3i].xyz;
        
        const float3 e1 = v2-v1;
        const float3 e2 = v3-v1;
        float3 normal = cross(e1, e2);
    
        ### if obj.shading != 'flat'
            __global const float4 *normal_data = vector_data + n_vertices;
        
            // compute barycentric coordinates
            const float3 f1 = v1 - pos;
            const float3 f2 = v2 - pos;
            
            const float normal_length = length(normal);
            const float3 tri_normal = normal / length(normal);
            
            const float a3 = dot(tri_normal, cross(e1,f1)) / normal_length;
            const float a2 = dot(tri_normal, cross(f1,e2)) / normal_length;
            const float a1 = dot(tri_normal, cross(v3-v2,f2)) / normal_length;
            
            const float3 n1 = normal_data[v1i].xyz;
            const float3 n2 = normal_data[v2i].xyz;
            const float3 n3 = normal_data[v3i].xyz;
            
            normal = n1*a1 + n2*a2 + n3*a3;
            
        ### endif
        *p_normal = fast_normalize(normal);
        
    ### endcall
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{obj.n_faces}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}}, {{obj.n_vertices}});
### endmacro
