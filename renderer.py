
from accelerator import Accelerator
import numpy as np
import utils
import itertools
from cl_compiler import Compiler

# pylint: disable-msg=W0201

class Renderer:
    """
    Main controller class that, when initialized with the Scene, generates
    and compiles the OpenCL code for rendering it and provides methods:
        
        * render_sample for sampling a new batch of rays and accumulating
          the results to an image
        * get_image for accessing the currently rendered result

    """
    # TODO: refactor to better separated classes
    
    def get_image(self):
        imgdata = self.img.get().astype(np.float32)
        img = np.empty(self.img_shape + (3,))
        img[self.image_order[:, 0], self.image_order[:, 1], :] = imgdata[..., 0:3]
        return img
        
    def rays_per_sample(self):
        return self.img_shape[0]*self.img_shape[1]
    
    def __init__(self, scene, args):
        
        self.shader = scene.shader(scene)
        
        self.scene = scene
        self._group_objects()
        self._init_lights()
        
        self.acc = Accelerator(args.choose_opencl_context)
        self._collect_tracer_data()
        self._init_misc()
        self._init_camera_and_image()
        self.ray_state = RayStateBuffers(self)
        self.shader.initialize_material_buffers(self.acc)
        
        program_code = Compiler(self).make_program()
        self.prog = self.acc.build_program(program_code, args.cl_build_options)
    
    def _group_objects(self):
        get_tracer_name = lambda obj: obj.tracer.tracer_kernel_name
        self.scene.objects.sort(key=get_tracer_name)
        
        groups = itertools.groupby(self.scene.objects, key=get_tracer_name)
        groups = [ (k,list(g)) for k,g in groups ]
        object_counts = { k : len(g) for k, g in groups }
        
        offset = 0
        self.object_groups = []
        for name in sorted([k for k,_ in groups]):
            
            count = object_counts[name]
            tracer = self.scene.objects[offset].tracer
            
            self.object_groups.append((tracer,count,offset))
            
            offset += count

        # ------------- Find root container object
        self.root_object_id = 0
        for i in range(len(self.scene.objects)):
            if self.scene.root_object == self.scene.objects[i]:
                self.root_object_id = i+1
    
    def _init_misc(self):
        # TODO: bad
        
        self.max_broadcast_vecs = 6
        self.vec_broadcast = self.acc.new_const_buffer(np.zeros((self.max_broadcast_vecs, 4)))
        self.vec_param_buf = np.zeros((self.max_broadcast_vecs, 4), dtype=np.float32)
        
        # Randomization init
        self.qdirs = utils.quasi_random_direction_sample(self.scene.samples_per_pixel)
        self.qdirs = np.random.permutation(self.qdirs)
        
        self.kernel_work_groups = {}
        self.zero_scratch = self.acc.new_local_buffer(1)
    
    def _init_camera_and_image(self):
        scene = self.scene
        
        cam = self.scene.get_camera_rays()
        self.rotmat = scene.get_camera_rotmat()
        fovx_rad = scene.camera_fov / 180.0 * np.pi
        self.pixel_angle = fovx_rad / scene.image_size[0]
        
        self.img_shape = scene.image_size[::-1]
        self.n_pixels = self.img_shape[0] * self.img_shape[1]
        
        self.image_order = self.get_image_order()
        
        cam = cam[self.image_order[:, 0], self.image_order[:, 1], 0:3]
        
        # Device buffers
        self.cam = self.acc.make_vec3_array(cam)
        self.img = self.acc.new_vec3_array((self.n_pixels, ))
    
    def _init_lights(self):
        
        self.bidirectional_light_ids = [i
            for i in range(len(self.scene.objects))
            if self.scene.objects[i].bidirectional_light]
            
        self.bidirectional = len(self.bidirectional_light_ids) > 0
    
    def get_light_point(self):
        light_id = self.bidirectional_light_ids[np.random.randint(len(self.bidirectional_light_ids))]
        light = self.scene.objects[light_id]
        return (light_id, light.tracer.surface_area()) + \
            light.tracer.random_surface_point_and_normal() + \
            light.tracer.center_and_min_sampling_distance()
    
    def _collect_tracer_data(self):
        
        data_items = ['vector', 'integer',
            'param_float3', 'param_int', 'param_float']
        
        data = { k : [] for k in data_items }
        data_sizes = { k : 0 for k in data_items }
        
        object_data_pointer_buffer = []
        
        for obj in self.scene.objects:
            
            if obj.tracer.has_data():
                cur_data = obj.tracer.get_data()
            else:
                cur_data = {}
                
            param_values_by_type = {}
            local_param_offsets = []
            
            parameter_types = obj.tracer.parameter_types()
            param_values = obj.tracer.parameter_values()
            
            for p_idx in range(len(parameter_types)):
                cl_type = parameter_types[p_idx]
                old = param_values_by_type.get(cl_type, [])
                local_param_offsets.append(len(old))
                param_values_by_type[cl_type] = old + [param_values[p_idx]]
            
            obj.tracer.local_param_offsets = local_param_offsets
            
            for cl_type, params in param_values_by_type.items():
                param_type = 'param_' + cl_type
                assert(param_type in data_items)
                cur_data[param_type] = np.array(params)
            
            offset_buffer = []
            
            for dtype in data_items:
                cur = cur_data.get(dtype)
                n_data = data_sizes[dtype]
                
                offset_buffer.append(n_data)
                
                if cur is not None:
                    if dtype in ['vector','param_float3']:
                        if cur.shape[1] != 3:
                            raise RuntimeError('invalid vector data shape')
                        n_data += cur.shape[0]
                    else:
                        cur = np.ravel(cur)
                        n_data += cur.size
                    
                    data[dtype].append(cur)
                    data_sizes[dtype] = n_data
            
            object_data_pointer_buffer.append(offset_buffer)
        
        self.tracer_data_buffers = []
        self.tracer_const_data_buffers = []
        
        for dtype in data_items:
            values = data[dtype]
            
            if dtype == 'param_int':
                self.object_data_pointer_buffer_offset = data_sizes[dtype]
                values += object_data_pointer_buffer
            
            if len(values) == 0: values = None
            else:
                if dtype in ['vector', 'param_float3']:
                    values = np.vstack(values)
                else:
                    values = np.concatenate(values)
                
                if dtype == 'vector':
                    values = self.acc.make_vec3_array(values)
                    buffers = self.tracer_data_buffers
                elif dtype == 'integer':
                    values = self.acc.to_device(values.astype(np.int32))
                    buffers = self.tracer_data_buffers
                elif dtype == 'param_float3':
                    values = self.acc.make_const_vec3_buffer(values)
                    buffers = self.tracer_const_data_buffers
                elif dtype == 'param_float':
                    values = self.acc.new_const_buffer(values)
                    buffers = self.tracer_const_data_buffers
                elif dtype == 'param_int':
                    values = self.acc.new_const_buffer(values, np.int32)
                    buffers = self.tracer_const_data_buffers
                else:
                    assert(False)
            
            buffers.append(values)

    # helpers
    
    def _fill_vec(self, data, vec):
        hostbuf = np.float32(vec)
        self.acc.enqueue_copy(self.vec_broadcast, hostbuf)
        self.acc.call('fill_vec_broadcast', self.n_pixels, (data, ), \
            value_args=(self.vec_broadcast, ))

    def new_camera_sample(self):
        
        scene = self.scene
        cam_origin = scene.camera_position
        
        # TODO: quasi random...
        sx = np.float32(np.random.rand())
        sy = np.float32(np.random.rand())
        
        # Tent filter as in smallpt
        if self.scene.tent_filter:
            def tent_filter_transformation(x):
                x *= 2
                if x < 1:
                    return np.sqrt(x)-1
                else:
                    return 1-np.sqrt(2-x)
            
            sx = tent_filter_transformation(sx)
            sy = tent_filter_transformation(sy)
        
        overlap = 0.0
        thetax = (sx-0.5)*self.pixel_angle*(1.0+overlap)
        thetay = (sy-0.5)*self.pixel_angle*(1.0+overlap)
        
        dofx, dofy = utils.random_dof_sample()
        
        dof_pos = (dofx * self.rotmat[:, 0] + dofy * self.rotmat[:, 1]) * scene.camera_dof_fstop
        
        sharp_distance = scene.camera_sharp_distance
        
        tilt = utils.rotmat_tilt_camera(thetax, thetay)
        mat = np.dot(np.dot(self.rotmat, tilt), self.rotmat.transpose())
        mat4 = np.zeros((4, 4))
        mat4[0:3, 0:3] = mat
        mat4[3, 0:3] = dof_pos
        mat4[3, 3] = sharp_distance
        
        cam_origin = cam_origin + dof_pos
        
        return cam_origin, mat4

    def render_sample(self, sample_index):
    
        scene = self.scene
        acc = self.acc
    
        cam_origin, mat4 = self.new_camera_sample()
        
        acc.enqueue_copy(self.vec_broadcast,  mat4.astype(np.float32))
        acc.call('subsample_transform_camera', self.n_pixels, \
            (self.cam, self.ray_state.ray,), \
            value_args=(self.vec_broadcast,))
        
        self._fill_vec(self.ray_state.pos, cam_origin)
        self.ray_state.whichobject.fill(0)
        self.ray_state.normal.fill(0)
        self.ray_state.raycolor.fill(1)
        
        self.ray_state.inside.fill(self.root_object_id)
        
        path_index = 0
        r_prob = 1
        
        self.shader.init_sample(self)
        
        while True:
            
            self.ray_state.raycolor *= r_prob
            
            r_prob = 1
            break_next = False
            if path_index >= scene.min_bounces:
                rand_01 = np.random.rand()
                if rand_01 < scene.russian_roulette_prob and path_index < scene.max_bounces:
                    r_prob = 1.0/(1-scene.russian_roulette_prob)
                else: break_next = True
            
            self.compute_next_path_segment(sample_index, path_index, break_next)
            
            if break_next:
                break
            
            path_index += 1
    
        acc.finish()
        
        return path_index

    def compute_next_path_segment(self, sample_index, path_index, is_last):
        
        acc = self.acc
        
        self.ray_state.isec_dist.fill(self.scene.max_ray_length)
        self.ray_state.last_whichobject = self.ray_state.whichobject * 1
        self.ray_state.last_which_subobject = self.ray_state.which_subobject * 1
        
        for tracer, count, offset in self.object_groups:
            self.call_grouped_kernel(tracer.tracer_kernel_name, count, \
                self.ray_state.tracer_kernel_params() + \
                tuple(self.tracer_data_buffers),
                value_args=tuple(self.tracer_const_data_buffers) + \
                    (np.int32(offset+1), np.int32(count)))
        
        for tracer, count, offset in self.object_groups:
            self.acc.call(tracer.normal_kernel_name, self.n_pixels, \
                    self.ray_state.normal_kernel_params() + \
                    tuple(self.tracer_data_buffers),
                    value_args=tuple(self.tracer_const_data_buffers) + \
                    (np.int32(offset+1), np.int32(count)))
        
        if self.bidirectional:
            if path_index == 0:
                self.ray_state.suppress_emission.fill(0)
        
            light_id0, light_area, light_point, light_normal, \
                light_center, min_light_sampling_distance = self.get_light_point()
            
            light_point = np.array(light_point).astype(np.float32)
            light_normal = np.array(light_normal).astype(np.float32)
            light_center = np.array(light_center).astype(np.float32)
            light_area = np.float32(light_area)
            min_light_sampling_distance = np.float32(min_light_sampling_distance)
                
            if is_last:
                light_id1 = np.int32(0)
            else:
                light_id1 = np.int32(light_id0+1)
                self.vec_param_buf[0, :3] = light_point

                acc.enqueue_copy(self.vec_broadcast, self.vec_param_buf)
                self.ray_state.shadow_mask.fill(1.0)
                
                
                for tracer, count, offset in self.object_groups:
                    self.call_grouped_kernel(tracer.shadow_kernel_name, count, \
                        self.ray_state.shadow_kernel_params() + \
                        tuple(self.tracer_data_buffers),
                        value_args=tuple(self.tracer_const_data_buffers) + \
                            (self.vec_broadcast, light_id1, np.int32(offset+1), np.int32(count)))
    
        if self.scene.quasirandom and path_index == 1:
            rand_vec = self.qdirs[sample_index, :]
        else:
            rand_vec = utils.normalize(np.random.normal(0, 1, (3, )))
            
        rand_vec = np.array(rand_vec).astype(np.float32) 
        rand_01 = np.float32(np.random.rand())
        
        self.vec_param_buf[0, :3] = rand_vec
        self.vec_param_buf[1, :3] = np.random.normal(0, 1,( 3, ))
        # element 2 has color mask
        if self.bidirectional:
            self.vec_param_buf[3, :3] = light_point
            self.vec_param_buf[4, :3] = light_normal
            self.vec_param_buf[5, :3] = light_center
        acc.enqueue_copy(self.vec_broadcast, self.vec_param_buf)
        
        constant_params = self.shader.material_buffers + [rand_01, self.vec_broadcast]
        if self.bidirectional:
            constant_params = [light_id1, light_area, min_light_sampling_distance] + constant_params
        
        acc.call(self.shader.shader_name, self.n_pixels, \
            (self.img, ) + self.ray_state.shader_kernel_params(),
            value_args=tuple(constant_params))
        
    def call_grouped_kernel(self, kernel_name, group_size, buffer_args, value_args):
        
        global_size, local_size, scratch_buffers = \
            self.get_work_group_sizes_and_scratch(kernel_name, group_size)
        
        self.acc.call(kernel_name, global_size, buffer_args,
            value_args + scratch_buffers, work_group_size=local_size)
    
    def get_image_order(self):
        order = []
        
        width, height = self.img_shape
        
        block_w = 8
        block_h = 4
        
        if width % block_w == 0 and height % block_h == 0:
            for y_block in range(height / block_h):
                for x_block in range(width / block_w):
                    for by in range(block_h):
                        for bx in range(block_w):
                            order.append([bx+x_block*block_w, by+y_block*block_h])
        else:
            print "WARNING: image size should be (%d*x, %d*y)" % (block_w, block_h)
        
            for x in range(width):
                for y in range(height):
                    order.append([x,y])
            
        return np.array(order)
        
    def get_work_group_sizes_and_scratch(self, kernel_name, group_size):
        
        cache = self.kernel_work_groups
        key = (kernel_name, group_size)
        
        if key in cache: return cache[key]
        
        if group_size == 1:
            local_size = None
            global_size = self.n_pixels
            float_scratch = self.zero_scratch
            int_scratch = self.zero_scratch
        else:
            #block_size = self.acc.get_preferred_local_work_group_size_multiple(kernel_name)
            max_size = self.acc.get_max_work_group_size(kernel_name)
            
            whole_groups = max_size / group_size
            assert(whole_groups > 0)
            local_size = (whole_groups, group_size)
            
            if self.n_pixels % whole_groups != 0:
                padding = whole_groups - (self.n_pixels % whole_groups)
                global_size = (self.n_pixels+padding, group_size)
            else:
                global_size = (self.n_pixels, group_size)
        
            print "work_group sizes for", key, "=", global_size, local_size
            
            n_scratch = local_size[0]*local_size[1]
            float_scratch = self.acc.new_local_buffer(n_scratch)
            int_scratch = self.acc.new_local_buffer(n_scratch)
        
        cache[key] = (global_size, local_size, (float_scratch, int_scratch))
        return cache[key]
        

class RayStateBuffers:
    def __init__(self, renderer):
        n_pixels = renderer.n_pixels
        acc = renderer.acc
        self.bidirectional = renderer.bidirectional
        
        self.whichobject = acc.new_array((n_pixels, ), np.uint32, True)
        self.which_subobject = acc.zeros_like(self.whichobject)
        self.last_which_object = acc.zeros_like(self.whichobject)
        self.last_which_subobject = acc.zeros_like(self.whichobject)
        self.pos = acc.new_vec3_array((n_pixels, ))
        self.ray = acc.zeros_like(self.pos)
        self.inside = acc.zeros_like(self.whichobject)
        self.normal = acc.zeros_like(self.pos)
        self.isec_dist = acc.zeros_like(self.pos)
        
        self.raycolor = renderer.shader.new_ray_color_buffer(acc, (n_pixels, ))
        
        if self.bidirectional:
            self.shadow_mask = acc.zeros_like(self.isec_dist)
            self.suppress_emission = acc.new_array((n_pixels, ), np.int32, True)
    
    def tracer_kernel_params(self):
        return (self.pos, self.ray, \
                self.isec_dist, self.whichobject, self.which_subobject, \
                self.last_whichobject, self.last_which_subobject,
                self.inside)
            
    def shadow_kernel_params(self):
        return (self.pos, self.normal, self.whichobject, \
                self.which_subobject, self.inside, \
                self.shadow_mask)

    def normal_kernel_params(self):
        return (self.pos, self.ray, \
                self.normal, self.isec_dist, self.whichobject, \
                self.which_subobject, self.inside)
    
    def shader_kernel_params(self):
        buffer_params = [self.whichobject, 
            self.normal, self.isec_dist, self.pos, self.ray,
            self.raycolor, self.inside]
        
        if self.bidirectional:
            buffer_params += [self.shadow_mask, self.suppress_emission]
        
        return tuple(buffer_params)
        