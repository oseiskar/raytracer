
def make_program( scene ):

    objects = [obj.tracer for obj in scene.objects]
    Nobjects = len(scene.objects)

    cl_utils = open('cl/utils.cl', 'r').read() # static code

    # ------------- make tracer kernel (finds intersections)
    trace_kernel = """
    __kernel void trace(
        __global float3 *p_pos,
        __global const float3 *p_ray,
        __global float3 *p_normal,
        __global float *p_isec_dist,
        __global uint *p_whichobject,
        __global const uint *p_inside)
    {
        const int gid = get_global_id(0);
        const float3 ray = p_ray[gid];
        float3 pos = p_pos[gid];
        const float3 last_normal = p_normal[gid];
        const uint lastwhichobject = p_whichobject[gid];
        const uint inside = p_inside[gid];
        
        p_whichobject += gid;
        p_normal += gid;
        p_isec_dist += gid;
        p_pos += gid;
        
        float old_isec_dist = *p_isec_dist;
        float new_isec_dist = 0;
        uint subobject;
        uint cur_subobject;
        
        uint i = 0;
        uint whichobject = 0;
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
                "lastwhichobject == i"])
        
        # TODO: handle non-hitting rays!
        
        trace_kernel += """
        if (//lastwhichobject != i && // cull self
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
        *p_isec_dist = old_isec_dist;
        *p_whichobject = whichobject;
        *p_pos = pos;
    }
    """

    # ------------- shader kernel

    shader_kernel_params = """
    #define RUSSIAN_ROULETTE_PROB %s
    """ % scene.russian_roulette_prob

    shader_kernel = shader_kernel_params
    shader_kernel += open('cl/' + scene.shader + '.cl').read()

    prog_code = cl_utils

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

    prog_code += "\n"

    prog_code += "\n".join(list(kernels))
    prog_code += trace_kernel
    prog_code += shader_kernel
    
    return prog_code
