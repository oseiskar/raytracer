
import numpy as np

class Shader:
    """
    The differences between RGB and Spectrum shading in the Renderer are
    abstracted behind this interface.
    """
    
    def __init__(self, scene):
        self.scene = scene
    
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
    
    def get_material_property_offsets(self):
        properties = []
        n_objects = len(self.scene.objects)
        
        for property_list in self.material_property_sets:
            
            offset = 0
            for p in property_list:
                properties.append((p.upper(), offset))
                offset += n_objects + 1
        
        return properties
    
    def init_sample(self, renderer):
        pass

class RgbShader(Shader):
    """
    In RGB shading mode, all color properties of materials are represented
    with RGB color values. This converges faster and the colors are easier
    to represent than in Spectrum mode, but there is no support for dispersion
    and the model is less physics-based.
    """
    
    def __init__(self, scene):
        Shader.__init__(self, scene)
        
        self.rgb = True
        
        self.material_property_sets = [
            # RGB material properties
            [
                'diffuse',
                'emission',
                'reflection',
                'transparency',
                'volume_absorption'
            ],
            # scalar material properties
            [
                'ior',
                'volume_scattering',
                'volume_scattering_blur',
                'reflection_blur',
                'transparency_blur'
            ]
        ]
    
    def new_ray_color_buffer(self, acc, shape): 
        return acc.new_vec3_array(shape)

    def initialize_material_buffers(self, acc):
        
        n_objects = len(self.scene.objects)
        p_buf_len = n_objects+1
        
        self.material_buffers = []
        
        for color in [True, False]:
            
            property_list = self.material_property_sets[int(not color)]
            
            if color:
                buf = np.zeros((p_buf_len*len(property_list), 4))
            else:
                buf = np.zeros((p_buf_len*len(property_list), 1))
                
            for obj_idx, p_idx, value in self.each_object_material(property_list):
                
                if np.array(value).size == 1:
                    if color:
                        value = [value]*3
                    else:
                        value = [value]
                value = np.array(value)
                if color and value.size == 1:
                    value = np.ones((3, ))*value[0]
                buf[p_idx * p_buf_len + obj_idx, :value.size] = value
            
            device_buffer = acc.new_const_buffer(buf)
            
            self.material_buffers.append(device_buffer)
    

class SpectrumShader(Shader):
    """
    In Spectrum shading mode, material colors are represented using spectra
    (e.g., emission and absorption spectra), which enables accurate color
    physics (e.g., black body radiation) and dispersion effects.
    """
    
    def __init__(self, scene):
        Shader.__init__(self, scene)
        
        self.rgb = False
        
        self.material_property_sets = [
            # (scalar) material properties
            [
                'diffuse',
                'emission',
                'reflection',
                'transparency',
                'volume_scattering',
                'volume_absorption',
                'reflection_blur',
                'transparency_blur',
                'volume_scattering_blur',
                'ior'
            ]
        ]
    
    def new_ray_color_buffer(self, acc, shape): 
        return acc.new_array(shape, np.float32, True)
    
    def initialize_material_buffers(self, acc):
        
        spectrum = self.scene.spectrum
        
        property_list = self.material_property_sets[0]
        self.host_material_properties = []
        
        self.color_responses = spectrum.cie_1931_rgb()
        host_mat_y = []
        
        self.color_intensity_pdf = spectrum.visible_intensity()
        self.color_intensity_cdf = np.cumsum(self.color_intensity_pdf)
            
        for _, __, value in self.each_object_material(property_list):
            
            y = np.ravel(np.array(value))
            x = np.linspace( *spectrum.wavelength_range, num=y.size )
            
            host_mat_y.append( spectrum.map_left(x, y) )
        
        self.host_mat_y = np.vstack(host_mat_y).astype(np.float32)
        self.device_material_buffer = acc.new_const_buffer(self.host_mat_y[:, 0])
        
        self.material_buffers = [self.device_material_buffer]
    
    def init_sample(self, renderer):
        
        spectrum = self.scene.spectrum
        
        # importance sampling
        omega = np.random.random()
        wavelength = np.interp([omega], self.color_intensity_cdf, spectrum.wavelengths)
        wavelength_prob = spectrum.map_right([wavelength], self.color_intensity_pdf)
        
        #print wavelength, wavelength_prob
        
        renderer.ray_state.raycolor *= 1.0 / wavelength_prob
        
        wave_interp = lambda y: \
            spectrum.map_right([wavelength], y).astype(np.float32)
        
        mat_props = wave_interp(self.host_mat_y)
        renderer.acc.enqueue_copy(self.device_material_buffer, mat_props)
        
        color_mask = wave_interp(self.color_responses)
        #print wavelength, color_mask
        renderer.vec_param_buf[2, :3] = np.ravel(color_mask)
    
