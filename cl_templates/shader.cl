
### for item in shader.get_material_property_offsets()
    #define MAT_{{item[0]}} {{item[1]}}
### endfor

### if shader.rgb
    #define COLOR(prop, id) material_colors[prop + id].xyz
    #define WHITE (float3)(1.0,1.0,1.0)
    #define COLOR2PROB(color) dot(color,WHITE)/3.0
### else
    #define COLOR(prop, id) material_scalars[prop + id]
    #define WHITE 1.0
    #define COLOR2PROB(color) color
### endif

#define SCALAR(prop, id) material_scalars[prop + id]

### macro shader_kernel(name)

__kernel void shader_{{name}}(
        // output: colors are summed to this image when appropriate
        global float3 *img,
        // probability sample for the current pixel
        global float *p_p,
        // number of diffusive bounces left for this ray
        global int *diffusions_left,
        // which image pixel this ray affects
        global int *pixel,
        // local color multiplier, everything added to the image
        // is multiplied by this and the ray color. Applied to ray_color
        // at the end of the pipeline
        global {{ shader.color_cl_type }} *pipeline_color,
        // id of the object on whose surface the ray position (param
        // pos below) is currently on. If 0, the ray position is not
        // on any surface (happens with fog/scattering and rays
        // starting from the camera)
        global uint *surface_object_id,
        // surface normal at ray position (if on a surface)
        global float3 *normal,
        // ...
        global float *last_distance,
        // current ray position
        global float3 *pos,
        // current ray direction
        global float3 *ray,
        // current ray color: a filter that multiplies everything
        // added to the image
        global {{ shader.color_cl_type }} *ray_color,
        // id of the object inside which the current ray position is,
        // 0 if in ambient space. (notice however, that the ambient
        // space also has fog and IoR material properties)
        global uint *inside,
### if renderer.bidirectional
        // visibility mask for light point
        global float *shadow_mask,
        global int *suppress_emission,
        // id of the light object
        int light_id,
        float light_area,
        float min_light_sampling_distance,
### endif
        // material properties of the current object
### if shader.rgb
        constant float4 *material_colors,
### endif
        constant float *material_scalars,
        // a random unit vector and a random gaussian vector
        constant float4 *rvecs_cmask_and_light)
{
    const int thread_idx = get_global_id(0);
    pixel += thread_idx;
    
    const int ray_idx = *pixel;
    //if (ray_idx < 0) return;
    
    p_p += thread_idx;
    pipeline_color += thread_idx;
    
    // the random sample [0,1) that decides what to do
    float p = *p_p;
    if (p < 0.0) return;
        
    ray += ray_idx;
    normal += thread_idx;
    last_distance += thread_idx;
    
    img += ray_idx;
    surface_object_id += ray_idx;
    pos += ray_idx;
    ray_color += ray_idx;
    inside += ray_idx;
    diffusions_left += ray_idx;
### if renderer.bidirectional
    suppress_emission += ray_idx;
    shadow_mask += thread_idx;
### endif
    
    float3 r = *ray;
    float3 n = *normal;
    const float3 rvec = rvecs_cmask_and_light[0].xyz;
    float3 gauss_rvec = rvecs_cmask_and_light[1].xyz;
    
    float cur_prob;
    {{ shader.color_cl_type }} cur_col;
    
### if shader.rgb
    const float cmask = 1.0;
### else
    const float3 cmask = rvecs_cmask_and_light[2].xyz;
### endif
    
    {{ caller() }}
    
    if (p < 0.0)
    {
        *ray = r;
        *ray_color *= *pipeline_color;
        // if (COLOR2PROB(*ray_color) == 0.0) *pixel = -1;
    }
    *p_p = p;
}
### endmacro

### call shader_kernel('volumetric')

    // -------------------- volumetric effects

    float last_dist = *last_distance;
    const float alpha = SCALAR(MAT_VOLUME_SCATTERING,*inside);
    if (alpha > 0) cur_prob = 1.0-exp(-alpha*last_dist);
    
    if (p < cur_prob)
    {
        // (sub-surface) scattering / fog
        
        // "derivation":
        //    p < 1.0-exp(-alpha*dist)
        //   (1-p) > exp(-alpha*dist)
        //   log(1-p) > -alpha*dist
        //   -log(1-p) < alpha*dist
        //   dist > -log(1-p)/alpha
        float d = -log(1.0 - p) / alpha;
      
        *pos -= (last_dist-d) * r;
        last_dist = d;
        *surface_object_id = 0; // not on any surface
        
        const float blur = SCALAR(MAT_VOLUME_SCATTERING_BLUR,*inside);
        if (blur < 1.0) {
            r = normalize(gauss_rvec + r * tan(M_PI*0.5*(1.0 - blur)));
        }
        else r = rvec;
        
        ### if renderer.bidirectional
            *suppress_emission = 0;
        ### endif
        
        p = -1.0;
    }
    else p -= cur_prob;
    
    // volumetric absorption
    ### if shader.rgb
        cur_col = COLOR(MAT_VOLUME_ABSORPTION,*inside);
        *pipeline_color = (float3)(exp(-cur_col.x*last_dist),exp(-cur_col.y*last_dist),exp(-cur_col.z*last_dist));
    ### else
        *pipeline_color = exp(-SCALAR(MAT_VOLUME_ABSORPTION,*inside)*last_dist);
    ### endif

### endcall

### call shader_kernel('emission')

    // -------------------- emission
    
    ### if renderer.bidirectional
        const int suppressed = *suppress_emission;
        
        if ((suppressed >= 0 && *surface_object_id != suppressed) || *surface_object_id == -suppressed)
        {
    ### endif
            *img += COLOR(MAT_EMISSION, *surface_object_id)
                * (*ray_color) * (*pipeline_color) * cmask;
    ### if renderer.bidirectional
        }
    ### endif

### endcall

### call shader_kernel('reflection')

    // -------------------- specular reflection
    cur_col = COLOR(MAT_REFLECTION, *surface_object_id);
    cur_prob = COLOR2PROB(cur_col);
    
    if (p < cur_prob)
    {
        ### if shader.rgb
            *pipeline_color *= cur_col / cur_prob; // 1 in spectrum shader
        ### endif
        
        r -= 2*dot(r,n) * n;
        
        const float blur = SCALAR(MAT_REFLECTION_BLUR, *surface_object_id);
        if (blur > 0) {
            if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
            r = normalize(gauss_rvec * blur + r * (1.0-blur));
        }
        
        ### if renderer.bidirectional
            *suppress_emission = 0;
        ### endif
        p = -1.0;
    }
    else p -= cur_prob;
    
### endcall

### call shader_kernel('refraction')

    // -------------------- refraction / transparency
    cur_col = COLOR(MAT_TRANSPARENCY, *surface_object_id);
    cur_prob = COLOR2PROB(cur_col);
    
    if (p < cur_prob)
    {
        ### if shader.rgb
            *pipeline_color *= cur_col / cur_prob; // 1 in spectrum shader
        ### endif
        
        float dotp = dot(r,n);
        if (dotp > 0) { n = -n; dotp = -dotp; }
                
        float nfrac = 0;
        
        float ior1 = SCALAR(MAT_IOR, *surface_object_id);
        float ior0 = SCALAR(MAT_IOR,0);
        
        if (*inside == *surface_object_id) // Leaving
        {
            nfrac = ior1/ior0;
            *inside = 0; // TODO: parent
        }
        else // going in
        {
            nfrac = ior0/ior1;
            *inside = *surface_object_id;
        }
        
        if (nfrac != 1)
        {
            float cos2t =1-nfrac*nfrac*(1-dotp*dotp);
            
            if (cos2t < 0)
            {
                // Total reflection
                
                r -= 2*dotp * n;
                
                if (*inside == *surface_object_id) *inside = 0;
                else *inside = *surface_object_id;
            }
            else
            {
                // Refraction
                r = normalize(r*nfrac + n*(-dotp*nfrac-sqrt(cos2t)));
            }
        }
        
        // else: no refraction, leave r intact
        
        if (dot(n,r) < 0) n = -n;
        *normal = n;
        
        const float blur = SCALAR(MAT_TRANSPARENCY_BLUR, *surface_object_id);
        if (blur > 0) {
            if (dot(n,gauss_rvec) < 0) gauss_rvec = -gauss_rvec;
            r = normalize(gauss_rvec * blur + r * (1.0-blur));
        }
        
        ### if renderer.bidirectional
            *suppress_emission = 0;
        ### endif
        p = -1.0;
    }
    else p -= cur_prob;

### endcall


### call shader_kernel('diffuse')

    ### if renderer.bidirectional
        *suppress_emission = 0;
    ### endif
    
    // -------------------- diffusion / absorbtion
    cur_col = COLOR(MAT_DIFFUSE, *surface_object_id);
    cur_prob = COLOR2PROB(cur_col);
    
    if (cur_prob > 0.0 && (*diffusions_left) > 0) {
        
        *pipeline_color *= cur_col;
        
        *diffusions_left -= 1;
    
        ### if renderer.bidirectional
            if (light_id > 0) {
                const float3 light_center = rvecs_cmask_and_light[5].xyz;
                
                if (length(*pos-light_center) > min_light_sampling_distance) {
                
                    const float3 light_point = rvecs_cmask_and_light[3].xyz;
                    float3 shadow_ray = light_point - *pos;
                    const float3 light_normal = rvecs_cmask_and_light[4].xyz;
                
                    *suppress_emission = light_id;
                    
                    if (dot(shadow_ray, n) > 0 && dot(shadow_ray,light_normal) < 0) {
                        const float shadow_dist = length(shadow_ray);
                        shadow_ray = fast_normalize(shadow_ray);
                        
                        *img += dot(n,shadow_ray) / M_PI
                            * dot(-shadow_ray,light_normal) / (shadow_dist*shadow_dist)
                            * (*ray_color) * (*pipeline_color) * cmask
                            * COLOR(MAT_EMISSION,light_id)
                            * (*shadow_mask)
                            * light_area;
                    }
                }
                else {
                    if (*diffusions_left == 0) {
                        *diffusions_left = 1;
                        *suppress_emission = -light_id;
                    }
                }
            }
        ### endif
    
        // diffuse reflection
        r = rvec;
            
        // Reflect (negation) to outside
        if (dot(n,r) < 0) r = -r;
        
        // Lambert's law
        *pipeline_color *= 2.0 * dot(n,r);
    }
    else *pipeline_color *= 0.0;
    
    p = -1.0;

### endcall

__kernel void init_shadow_mask(
    global const int *pixel,
    global float *shadow_mask,
    global const int *diffusions_left)
{
    const int thread_idx = get_global_id(0);
    const int ray_idx = pixel[thread_idx];
    
    if (diffusions_left[ray_idx] < 1) {
        shadow_mask[thread_idx] = 0.0;
    }
    else {
        shadow_mask[thread_idx] = 1.0;
    }
}

__kernel void culler(
        // which image pixel this ray affects
        global int *pixel,
        // current ray color: a filter that multiplies everything
        // added to the image
        global {{ shader.color_cl_type }} *ray_color,
        // probability sample for Russian roulette
        float russian_roulette_sample)
{
    const int thread_idx = get_global_id(0);
    pixel += thread_idx;
    
    const int ray_idx = *pixel;
    ray_color += ray_idx;
    
    const int local_idx = get_local_id(0);
    
    const float cur_intensity = COLOR2PROB(*ray_color);
    
    local float scratch[{{renderer.warp_size}}];
    
    float max_value;
    scratch[local_idx] = cur_intensity;
    
    ### for itr in range(renderer.log_warp_size)
    
        ### set sz = renderer.warp_size // (2**(itr+1))
    
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_idx < {{sz}})
            max_value = max(scratch[local_idx], scratch[local_idx + {{sz}}]);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (local_idx < {{sz}})
            scratch[local_idx] = max_value;
    
    ### endfor
    
    ### if renderer.scene.min_russian_prob
        local float russian_prob;
        
        if (local_idx == 0) {
            if (max_value > 0.0) {
                russian_prob = clamp(max_value, (float){{renderer.scene.min_russian_prob}}, (float)1.0);
                if (russian_roulette_sample > russian_prob)
                    scratch[0] = 0.0;
            }
            else russian_prob = 1.0;
        }
    ### endif
    
    barrier(CLK_LOCAL_MEM_FENCE);
    max_value = scratch[0];
    
    if (max_value == 0.0) {
        *pixel = -1;
    }
    ### if renderer.scene.min_russian_prob
        else if (russian_prob < 1.0) {
            *ray_color /= russian_prob;
        }
    ### endif
}
