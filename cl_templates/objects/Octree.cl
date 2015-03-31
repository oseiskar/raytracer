
 
// adapted from http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

### extends 'object.cl'

### import 'objects/TriangleMesh.cl' as triangle_mesh

### macro vec_less_than(target_var, source_var, threshold)
    // (py)opencl's vector < does seem to work as expected...
    ### for coord in ['x','y','z']:
        if ({{source_var}}.{{coord}} < {{threshold}}) {{target_var}}.{{coord}} = 1; 
    ### endfor
### endmacro

### macro coords_to_child_idx(target_var, coord_var)
    {{target_var}} = 0;
    if ({{coord_var}}.x > 0) {{target_var}} += 4;
    if ({{coord_var}}.y > 0) {{target_var}} += 2;
    if ({{coord_var}}.z > 0) {{target_var}} += 1;
### endmacro

### macro coords_inside_box(coord_var)
    {{coord_var}}.x >= 0 && {{coord_var}}.x <= 1 &&
    {{coord_var}}.y >= 0 && {{coord_var}}.y <= 1 &&
    {{coord_var}}.z >= 0 && {{coord_var}}.z <= 1
### endmacro

### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'uint root_data_offset, uint face_data_length, float3 root_origin, float root_size')
    
        // intersection with cube
        float3 slopes = 1.0 / ray;
        
        float3 rel = root_origin - origin;
        float3 walls1 = (float3)(0,0,0);
        {{ vec_less_than('walls1', 'slopes', '0.0') }}
        float3 walls2 = 1.0 - walls1;
        
        float3 isec1 = slopes * (rel + walls1*root_size), isec2 = slopes * (rel + walls2*root_size);
        
        float isec_begin = max(max(isec1.x,isec1.y),isec1.z);
        float isec_end = min(min(isec2.x,isec2.y),isec2.z);
    
        if (isec_begin > isec_end || isec_end < 0.0 || isec_begin > old_isec_dist) return;
        
        float3 coords;
        if (isec_begin < 0.0) coords = rel;
        else coords = rel + isec_begin*ray;
        coords = coords / root_size;
        
        __global int *octree_data = integer_data + face_data_length;
        
        __global int *octree_node = octree_data + root_data_offset;
        int child_mask = octree_node[0];
        int data_ptr = octree_node[1];
        
        uint old_subobject = *p_subobject, subobject;
        float isec_dist = min(isec_end, old_isec_dist);
        
        if (child_mask == 0) {
            
            int n_triangles = octree_data[data_ptr];
            data_ptr += 1;
            
            ### call(triangle_index) triangle_mesh.tracer_function_core()
                octree_data[data_ptr + {{triangle_index}}]
            ### endcall
            
            if (isec_dist < isec_end) {
                *p_new_isec_dist = isec_dist;
                *p_subobject = subobject;
            }
        }
        
        //*p_new_isec_dist = isec_begin;
    
    ### endcall
### endmacro

### macro normal_function(obj)
    {{ triangle_mesh.normal_function(obj) }}
### endmacro

### macro tracer_call(obj, params)
{{ obj.tracer_function_name }}({{params}}, {{obj.root_data_offset}}, {{obj.total_faces*3}}, {{vec3(obj.root.origin)}}, {{obj.root.size}});
### endmacro

### macro normal_call(obj, params)
{{ obj.normal_function_name }}({{params}});
### endmacro
