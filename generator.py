
# note to self: do not rename this file (or anything else for that matter) to
# "compiler.py", this would break numpy.load :(

def make_program( shader ):
    
    scene = shader.scene

    objects = [obj.tracer for obj in scene.objects]
    Nobjects = len(scene.objects)

    # ------------- make tracer kernel (finds intersections)
    trace_kernel = """
    
    uint trace_core(
        float3 ray,
        const float3 last_normal,
        const uint last_whichobject,
        const uint inside,
        __private float3 *p_pos,
        __private uint *p_subobject,
        __private float *p_isec_dist)
    {
    
        float3 pos = *p_pos;
        float old_isec_dist = *p_isec_dist;
        uint whichobject = 0, subobject, cur_subobject, i;
        
        float new_isec_dist = 0;
    """
    
    # Unroll loop to CL code
    for i in range(Nobjects):
        
        obj = objects[i]
        
        trace_kernel += """
        new_isec_dist = 0;
        i = %s;
        
        // call tracer
        """ % (i+1)
        
        trace_kernel += obj.make_tracer_call([ \
                "pos",
                "ray",
                "last_normal",
                "old_isec_dist",
                "&new_isec_dist",
                "&cur_subobject",
                "inside == i",
                "last_whichobject == i"])
        
        # TODO: handle non-hitting rays!
        
        trace_kernel += """
        if (//last_whichobject != i && // cull self
            new_isec_dist > 0 &&
            new_isec_dist < old_isec_dist)
        {
            old_isec_dist = new_isec_dist;
            whichobject = i;
            subobject = cur_subobject;
        }
        """

    trace_kernel += """
        pos += old_isec_dist * ray; // saxpy
    """
    
    trace_kernel += """
        
        *p_pos = pos;
        *p_subobject = subobject;
        *p_isec_dist = old_isec_dist;
        
        return whichobject;
    }
    
    __kernel void shadow_trace(
        __global const float3 *p_pos,
        __global const float3 *p_normal,
        __global const uint *p_whichobject,
        __global const uint *p_inside,
        __global float *p_shadow_mask,
        // a random unit vector and color mask
        constant float4 *p_dest_point)
    {
        const int gid = get_global_id(0);
        
        const float3 dest = p_dest_point[0].xyz;
        float3 pos = p_pos[gid];
        float shadow_dist = length(dest - pos);
        const float3 ray = fast_normalize(dest - pos);
        const float3 last_normal = p_normal[gid];
        
        if ( dot(last_normal, ray) < 0.0 ) {
            // TODO
            p_shadow_mask[gid] = 0.0;
            return;
        }
        
        float isec_dist = shadow_dist;
        uint subobject;
        
        trace_core(
            ray,
            last_normal,
            p_whichobject[gid],
            p_inside[gid],
            &pos,&subobject,&isec_dist);
        
        const float EPS = 1e-5; // TODO: get rid of epsilon
        
        if (isec_dist > shadow_dist - EPS) {
            // no shadow
            p_shadow_mask[gid] = 1.0;
        }
        else {
            // shadow
            p_shadow_mask[gid] = 0.0;
        }
    }
    
    __kernel void trace(
        __global float3 *p_pos,
        __global const float3 *p_ray,
        __global float3 *p_normal,
        __global float *p_isec_dist,
        __global uint *p_whichobject,
        __global const uint *p_inside)
    {
        const int gid = get_global_id(0);
        
        p_whichobject += gid;
        p_normal += gid;
        p_isec_dist += gid;
        p_pos += gid;
        
        const float3 ray = p_ray[gid];
        const uint inside = p_inside[gid];
        
        float3 pos = *p_pos;
        float isec_dist = *p_isec_dist;
        uint subobject;
        
        uint whichobject = trace_core(
            ray,
            *p_normal,
            *p_whichobject,
            inside,
            &pos,&subobject,&isec_dist);
        
        uint i = 0;
    """


    # whichobject
    # subobject
    # pos 
    # old_isec_dist
        

    for i in range(Nobjects):
        
        obj = objects[i]
        
        trace_kernel += """
        i = %s;
        """ % (i+1)
        
        trace_kernel += """
        if (whichobject == i)
        {
            // call normal
            %s
            if (inside == i) *p_normal = -*p_normal;
        }
        """ % obj.make_normal_call(["pos", "subobject", "p_normal"])

    trace_kernel += """
        *p_isec_dist = isec_dist;
        *p_whichobject = whichobject;
        *p_pos = pos;
    }
    """

    # static code
    
    prog_code = ''
    with open('cl/utils.cl', 'r') as f: prog_code += f.read() 
    #with open('cl/poly_solvers.cl', 'r') as f: prog_code += f.read()
        
    prog_code += "\n"
    prog_code += shader.make_code()

    kernel_map = {}
    for obj in scene.objects:
        for (k,v) in obj.tracer.make_functions().items():
            if k in kernel_map and kernel_map[k] != v:
                raise "kernel name clash!!"
            kernel_map[k] = v
    kernels = set(kernel_map.values())

    for kernel in kernels:
        curl = kernel.find('{')
        declaration = kernel[:curl] + ';\n'
        prog_code += declaration

    prog_code += "\n".join(list(kernels))
    prog_code += trace_kernel
    
    return prog_code
