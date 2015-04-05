
 
// adapted from http://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

### extends 'object.cl'

### import 'objects/TriangleMesh.cl' as triangle_mesh

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

 
### macro vec_less_than(target_var, source_var, threshold)
    // (py)opencl's vector < does seem to work as expected...
    {{target_var}} = (int3)(0,0,0);
    ### for coord in ['x','y','z']:
        if ({{source_var}}.{{coord}} < {{threshold}}) {{target_var}}.{{coord}} = 1; 
    ### endfor
### endmacro


### macro tracer_function(obj)
    ### call tracer_function_base(obj, 'uint root_data_offset, uint face_data_length, float3 root_origin, float root_size')
    
        // intersection with cube
        const float3 slopes = 1.0 / ray;
        int3 walls1;
        {{vec_less_than('walls1', 'slopes', '0.0') }}
        const int3 walls2 = 1 - walls1;
        const int3 slope_signs = 1 - 2*walls1;
        
        const float3 rel = root_origin - origin;
        float3 isec1 = slopes * (rel + convert_float3(walls1)*root_size),
               isec2 = slopes * (rel + convert_float3(walls2)*root_size);
        
        float isec_begin = max(max(isec1.x,isec1.y),isec1.z);
        float isec_end = min(min(isec2.x,isec2.y),isec2.z);
    
        if (isec_begin > isec_end || isec_end < 0.0 || isec_begin > old_isec_dist) return;
        
        __global const int *octree_data = integer_data + face_data_length;
        
        #ifndef MAX_OCTREE_DEPTH
        #define MAX_OCTREE_DEPTH 5
        #endif
        
        #ifndef MAX_TOTAL_ITR
        #define MAX_TOTAL_ITR 100
        #endif
        
        int3 octree_coords = (int3)(0,0,0);
        int3 local_coords = octree_coords;
        int path[MAX_OCTREE_DEPTH];
        
        path[0] = root_data_offset;
        uint depth = 0;
        
        uint old_subobject = *p_subobject, subobject;
        float node_size = root_size;
        
        for(int itr=0; itr < MAX_TOTAL_ITR; itr++) {
            
            
            float3 ray_pos = origin + isec_begin * ray, node_origin;
            int child_mask, data_ptr, child_idx;
            
            // find octree leaf
            while (depth < MAX_OCTREE_DEPTH-1) {
                
                child_mask = octree_data[path[depth]];
                data_ptr = octree_data[path[depth]+1];
                node_origin = root_origin + convert_float3(octree_coords)*node_size;
                
                // 0x100 is a special marker for an empty leaf
                if (child_mask == 0 || child_mask == 0x100) break; 
                
                const float3 rel_coords = ray_pos - node_origin;
                
                {{ vec_less_than('local_coords', 'rel_coords', '0.5*node_size') }}
                local_coords = 1 - local_coords;
                
                {{ coords_to_child_idx('child_idx', 'local_coords') }}
                path[depth+1] = data_ptr + child_idx*2;
                octree_coords = (octree_coords << 1) | local_coords;
                
                node_size *= 0.5;
                depth++;
            }
            
            isec2 = slopes * (node_origin - origin + convert_float3(walls2)*node_size);
            isec_end = min(min(isec2.x,isec2.y),isec2.z);
            
            if (child_mask != 0x100) {
                // triangle intersections in the current leaf cube
                int n_triangles = octree_data[data_ptr];
                
                if (n_triangles > 0) {
                    
                    float isec_dist = isec_end;
                
                    ### call(triangle_index) triangle_mesh.tracer_function_core()
                        octree_data[data_ptr + 1 + {{triangle_index}}]
                    ### endcall
                    
                    if (isec_dist < isec_end) {
                        *p_new_isec_dist = isec_dist;
                        *p_subobject = subobject;
                        return;
                    }
                    
                }
            }
            
            //int3 coord_hop = (isec_end == isec2)*slope_signs; // does not seem to work either
            int3 coord_hop = (int3)(0,0,0);
            
            if (isec_end == isec2.x) {
                coord_hop.x = slope_signs.x;
            } else if (isec_end == isec2.y) {
                coord_hop.y = slope_signs.y;
            } else if (isec_end == isec2.z) {
                coord_hop.z = slope_signs.z;
            }
            
            // travel up the current tree branch as long as necessary
            while (depth > 0) {
                local_coords += coord_hop;
                
                if ({{coords_inside_box('local_coords')}}) break;
                else {
                    depth--;
                    octree_coords = octree_coords >> 1;
                    local_coords = octree_coords & 0x1;
                    node_size *= 2.0;
                }
            }
            if (depth == 0) return; // exited root cube
            
            // move to sibling node
            octree_coords = (octree_coords & (~0x1)) | local_coords;
            
            {{ coords_to_child_idx('child_idx', 'local_coords') }}
            data_ptr = octree_data[path[depth-1]+1];
            path[depth] = data_ptr + child_idx*2;
            
            isec_begin = isec_end;
        }
    
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
