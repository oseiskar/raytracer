
from utils import *

class Shader:
    
    def __init__(self, scene, prog, acc):
        
        self.scene = scene
        self.acc = acc
        self.prog = prog
        
        self.rgb_shader = scene.shader == 'rgb_shader'
        
        # ------------- Set up camera
        
        cam = scene.get_camera_rays()
        self.rotmat = scene.get_camera_rotmat()
        fovx_rad = scene.camera_fov / 180.0 * np.pi
        self.pixel_angle = fovx_rad / scene.image_size[0]

        # ------------- Parameter arrays

        self.mat_diffuse = self.new_mat_buf('diffuse')
        self.mat_emission = self.new_mat_buf('emission')
        self.mat_reflection = self.new_mat_buf('reflection')
        self.mat_transparency = self.new_mat_buf('transparency')
        self.mat_vs = self.new_mat_buf('vs')
        self.mat_ior = self.new_mat_buf('ior')
        self.mat_dispersion = self.new_mat_buf('dispersion')
        self.max_broadcast_vecs = 4
        self.vec_broadcast = acc.new_const_buffer(np.zeros((self.max_broadcast_vecs,4)))

        self.cam = acc.make_vec3_array(cam)
        imgshape = scene.image_size[::-1]

        # Randomization init
        self.qdirs = quasi_random_direction_sample(scene.samples_per_pixel)
        self.qdirs = np.random.permutation(self.qdirs)

        # Device buffers. 
        self.img = acc.new_vec3_array(imgshape)
        self.whichobject = acc.new_array(imgshape, np.uint32, True)
        self.pos = acc.zeros_like(self.cam)
        self.ray = acc.zeros_like(self.pos)
        self.inside = acc.zeros_like(self.whichobject)
        self.normal = acc.zeros_like(self.pos)
        self.isec_dist = acc.zeros_like(self.img)
        
        if self.rgb_shader: self.raycolor = acc.zeros_like(self.img)
        else: self.raycolor = acc.new_array(imgshape, np.float32, True )
        
        self.curcolor = acc.zeros_like(self.raycolor)
        
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

    def new_mat_buf(self, pname):
        
        object_materials = [obj.material for obj in self.scene.objects]
        
        default = self.scene.materials['default'][pname]
        if len(default) > 1:
            w = 4
            default += (0,)
        else:
            w = 1
            
        Nobjects = len(self.scene.objects)
        buf = np.zeros((Nobjects+1,w))
        buf[0] = np.array(default)
        for i in range(Nobjects):
            if pname in self.scene.materials[object_materials[i]]:
                prop = self.scene.materials[object_materials[i]][pname]
                if w > 1: prop += (0,)
            else:
                prop = default
            buf[i+1] = np.array(prop)
            
        return self.acc.new_const_buffer(buf)
            
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
        while True:
            
            self.raycolor *= r_prob
            
            self.isec_dist.fill(scene.max_ray_length)
            acc.call('trace', (self.pos,self.ray,self.normal,self.isec_dist,self.whichobject,self.inside))
            
            if scene.quasirandom and k == 1:
                vec = self.qdirs[sample_index,:]
            else:
                vec = normalize(np.random.normal(0,1,(3,)))
                
            vec = np.array(vec).astype(np.float32) 
            rand_01 = np.float32(np.random.rand())
            
            hostbuf = np.zeros((3,4), dtype=np.float32)
            hostbuf[0,:3] = vec
            
            if not self.rgb_shader:
                
                wavelength = np.float32(np.random.rand())
                def tent_func(x):
                    if abs(x) > 1.0: return 0
                    return 1.0 - abs(x)
                
                color_mask = [tent_func( 4.0 * (wavelength-c) ) for c in [0.25,0.5,0.75]]
                
                hostbuf[1,:3] = color_mask
                dispersion_coeff = np.float32((wavelength - 0.5) * 2.0)
                
                #print wavelength, color_mask, dispersion_coeff
            
            acc.enqueue_copy(self.vec_broadcast, hostbuf)
            
            device_bufs_params = (\
                self.img, self.whichobject, self.normal,
                self.isec_dist, self.pos, self.ray, self.raycolor,
                self.inside)
            
            other_params = (\
                self.mat_emission,
                self.mat_diffuse,
                self.mat_reflection,
                self.mat_transparency,
                self.mat_ior)
            
            if self.rgb_shader:
                other_params += (self.mat_vs, rand_01, self.vec_broadcast)
            else:
                other_params += (self.mat_dispersion, self.mat_vs,
                    rand_01, dispersion_coeff, self.vec_broadcast)
            
            acc.call(scene.shader, device_bufs_params, other_params)
            
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


