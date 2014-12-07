
from utils import *
import generator
from accelerator import Accelerator

class Shader:
    
    def initialize(self, scene, args):
        
        self.scene = scene
        
        self.acc = Accelerator(scene.get_number_of_camera_rays(), \
            args.interactive_opencl_context)
        
        prog_code = generator.make_program(self)
        self.prog = self.acc.build_program( prog_code )
        
        self.prepare()
    
    def make_code(self):
        """main shader code generation function to be called by the generator"""
        
        code = self.make_definitions()
        with open('cl/' + self.shader_name + '.cl') as f: code += f.read()
        return code
    
    def make_definitions(self):
        
        code = ''
        n_objects = len(self.scene.objects)
        
        for property_list in self.material_property_sets:
            
            offset = 0
            
            for p in property_list:
                code += "#define MAT_%s %d\n" % (p.upper(), offset)
                offset += n_objects + 1
            
        return code
    
    def prepare(self):
        
        scene = self.scene
        
        # ------------- Set up camera
        
        cam = scene.get_camera_rays()
        self.rotmat = scene.get_camera_rotmat()
        fovx_rad = scene.camera_fov / 180.0 * np.pi
        self.pixel_angle = fovx_rad / scene.image_size[0]

        # ------------- Parameter arrays
        
        self.initialize_material_buffers()

        self.max_broadcast_vecs = 4
        self.vec_broadcast = self.acc.new_const_buffer(np.zeros((self.max_broadcast_vecs,4)))
        self.vec_param_buf = np.zeros((self.max_broadcast_vecs,4), dtype=np.float32)

        self.cam = self.acc.make_vec3_array(cam)
        imgshape = scene.image_size[::-1]

        # Randomization init
        self.qdirs = quasi_random_direction_sample(scene.samples_per_pixel)
        self.qdirs = np.random.permutation(self.qdirs)

        # Device buffers. 
        self.img = self.acc.new_vec3_array(imgshape)
        self.whichobject = self.acc.new_array(imgshape, np.uint32, True)
        self.pos = self.acc.zeros_like(self.cam)
        self.ray = self.acc.zeros_like(self.pos)
        self.inside = self.acc.zeros_like(self.whichobject)
        self.normal = self.acc.zeros_like(self.pos)
        self.isec_dist = self.acc.zeros_like(self.img)
        
        self.raycolor = self.new_ray_color_buffer(imgshape)
        self.curcolor = self.acc.zeros_like(self.raycolor)
        
        # ------------- Find root container object
        self.root_object_id = 0
        for i in range(len(scene.objects)):
            if scene.root_object == scene.objects[i]:
                self.root_object_id = i+1
        
    # helpers

    #def memcpy(self,dst,src): self.acc.enqueue_copy(dst.data, src.data)
    
    def fill_vec(self,data, vec):
        hostbuf = np.float32(vec)
        self.acc.enqueue_copy(self.vec_broadcast, hostbuf)
        self.acc.call('fill_vec_broadcast', (data,), (self.vec_broadcast,))

    def each_object_material(self, property_list):
        
        object_materials = [obj.material for obj in self.scene.objects]
        
        for p_idx in range(len(property_list)):
            pname = property_list[p_idx]
            default = np.array(self.scene.materials['default'][pname])
            
            yield((0, p_idx, default))
            
            for i in range(len(object_materials)):
                if pname in self.scene.materials[object_materials[i]]:
                    prop = self.scene.materials[object_materials[i]][pname]
                else:
                    prop = default
                
                yield((i+1, p_idx, prop))

    def extra_stuff(self): pass

    def render_sample(self,sample_index):
    
        scene = self.scene
        acc = self.acc
    
        cam_origin = scene.camera_position
        
        # TODO: quasi random...
        sx = np.float32(np.random.rand())
        sy = np.float32(np.random.rand())
        
        # Tent filter as in smallpt
        if self.scene.tent_filter:
            def tent_filter_transformation(x):
                x *= 2
                if x < 1: return np.sqrt(x)-1
                else: return 1-np.sqrt(2-x)
            
            sx = tent_filter_transformation(sx)
            sy = tent_filter_transformation(sy)
        
        overlap = 0.0
        thetax = (sx-0.5)*self.pixel_angle*(1.0+overlap)
        thetay = (sy-0.5)*self.pixel_angle*(1.0+overlap)
        
        dofx, dofy = random_dof_sample()
        
        dof_pos = (dofx * self.rotmat[:,0] + dofy * self.rotmat[:,1]) * scene.camera_dof_fstop
        
        sharp_distance = scene.camera_sharp_distance
        
        tilt = rotmat_tilt_camera(thetax,thetay)
        mat = np.dot(np.dot(self.rotmat,tilt),self.rotmat.transpose())
        mat4 = np.zeros((4,4))
        mat4[0:3,0:3] = mat
        mat4[3,0:3] = dof_pos
        mat4[3,3] = sharp_distance
        
        cam_origin = cam_origin + dof_pos
        
        acc.enqueue_copy(self.vec_broadcast,  mat4.astype(np.float32))
        acc.call('subsample_transform_camera', (self.cam,self.ray,), (self.vec_broadcast,))
        
        self.fill_vec(self.pos, cam_origin)
        self.whichobject.fill(0)
        self.normal.fill(0)
        self.raycolor.fill(1)
        self.curcolor.fill(0)
        kbegin = 0
        
        self.inside.fill(self.root_object_id)
        self.isec_dist.fill(0) # TODO
        
        k = kbegin
        r_prob = 1
        
        self.extra_stuff()
        
        while True:
            
            self.raycolor *= r_prob
            
            self.isec_dist.fill(scene.max_ray_length)
            acc.call('trace', (self.pos,self.ray,self.normal,self.isec_dist,self.whichobject,self.inside))
            
            if scene.quasirandom and k == 1:
                rand_vec = self.qdirs[sample_index,:]
            else:
                rand_vec = normalize(np.random.normal(0,1,(3,)))
                
            rand_vec = np.array(rand_vec).astype(np.float32) 
            rand_01 = np.float32(np.random.rand())
            
            self.vec_param_buf[0,:3] = rand_vec
            
            acc.enqueue_copy(self.vec_broadcast, self.vec_param_buf)
            
            acc.call(self.shader_name, \
                (self.img, self.whichobject, self.normal, self.isec_dist,
                 self.pos, self.ray, self.raycolor, self.inside), \
                tuple( self.material_buffers + [rand_01, self.vec_broadcast] ))
            
            r_prob = 1
            if k >= scene.min_bounces:
                rand_01 = np.random.rand()
                if rand_01 < scene.russian_roulette_prob and k < scene.max_bounces:
                    r_prob = 1.0/(1-scene.russian_roulette_prob)
                else:
                    break
            
            k += 1
    
        acc.finish()
        
        return k

        self.mat_diffuse = self.new_mat_buf('diffuse')
        self.mat_emission = self.new_mat_buf('emission')
        self.mat_reflection = self.new_mat_buf('reflection')
        self.mat_transparency = self.new_mat_buf('transparency')
        self.mat_vs = self.new_mat_buf('vs')
        self.mat_ior = self.new_mat_buf('ior')
        self.mat_dispersion = self.new_mat_buf('dispersion')

class RgbShader(Shader):
    
    def __init__(self, scene, args):
        self.shader_name = 'rgb_shader'
        
        self.material_property_sets = [
            # RGB material properties
            [
                'diffuse',
                'emission',
                'reflection',
                'transparency',
                'vs'
            ],
            # scalar material properties
            [ 'ior' ]
        ]
        
        self.initialize(scene, args)
    
    def new_ray_color_buffer(self, shape): 
        return self.acc.new_vec3_array(shape)

    def initialize_material_buffers(self):
        
        n_objects = len(self.scene.objects)
        p_buf_len = n_objects+1
        
        self.material_buffers = []
        
        for color in [True, False]:
            
            property_list = self.material_property_sets[int(not color)]
            
            if color: buf = np.zeros((p_buf_len*len(property_list),4))
            else: buf = np.zeros((p_buf_len*len(property_list),1))
                
            for obj_idx, p_idx, value in self.each_object_material(property_list):
                
                value = np.array(value)
                buf[p_idx * p_buf_len + obj_idx,:value.size] = value
            
            device_buffer = self.acc.new_const_buffer(buf)
            
            self.material_buffers.append(device_buffer)
    

class SpectrumShader(Shader):
    
    def __init__(self, scene, args):
        self.shader_name = 'spectrum_shader'
        
        self.material_property_sets = [
            # (scalar) material properties
            [
                'diffuse',
                'emission',
                'reflection',
                'transparency',
                'vs',
                'ior'
            ]
        ]
        self.initialize(scene, args)
    
    def new_ray_color_buffer(self, shape): 
        return self.acc.new_array(shape, np.float32, True)
    
    def initialize_material_buffers(self):
        
        n_objects = len(self.scene.objects)
        p_buf_len = n_objects+1
        
        spectrum = self.scene.spectrum
        
        property_list = self.material_property_sets[0]
        self.host_material_properties = []
        
        self.color_responses = spectrum.cie_1931_rgb()
        host_mat_y = []
        
        self.color_intensity_pdf = spectrum.visible_intensity()
        self.color_intensity_cdf = np.cumsum(self.color_intensity_pdf)
            
        for obj_idx, p_idx, value in self.each_object_material(property_list):
            
            y = np.ravel(np.array(value))
            x = np.linspace( *spectrum.wavelength_range, num=y.size )
            
            host_mat_y.append( spectrum.map_left(x,y) )
        
        self.host_mat_y = np.vstack(host_mat_y).astype(np.float32)
        self.device_material_buffer = self.acc.new_const_buffer(self.host_mat_y[:,0])
        
        self.material_buffers = [self.device_material_buffer]
    
    def extra_stuff(self):
        
        spectrum = self.scene.spectrum
        
        # importance sampling
        omega = np.random.random()
        wavelength = np.interp([omega], self.color_intensity_cdf, spectrum.wavelengths)
        wavelength_prob = spectrum.map_right([wavelength], self.color_intensity_pdf)
        
        #print wavelength, wavelength_prob
        
        self.raycolor *= 1.0 / wavelength_prob
        
        wave_interp = lambda y: \
            spectrum.map_right([wavelength],y).astype(np.float32)
        
        mat_props = wave_interp(self.host_mat_y)
        self.acc.enqueue_copy(self.device_material_buffer, mat_props)
        
        color_mask = wave_interp(self.color_responses)
        #print wavelength, color_mask
        self.vec_param_buf[1,:3] = np.ravel(color_mask)

