
from objects import *
import utils
import numpy as np
import shader
from spectrum import Spectrum

class Object:
    """
    An object consists of a Tracer that represents its shape and a material.
    It is also defined here if the object is to be used as a light source
    in bidirectional path tracing. If yes, this must be supported by the
    Tracer.
    """
    def __init__(self, tracer, material, name=None):
        self.tracer = tracer
        self.material = material
        self.name = name
        self.bidirectional_light = False

class Scene:
    
    def get_camera_rotmat(self):
        return utils.camera_rotmat(self.camera_direction, self.camera_up)
    
    def get_objects(self, name):
        return [obj for obj in self.objects if obj.name == name]
    
    def get_object(self, name):
        objs = self.get_objects(name)
        if len(objs) == 1: return objs[0]
        elif len(objs) == 0:
            raise KeyError("No object named '%s'" % name)
        else:
            raise KeyError("Multiple objects in the scene are called '%s'" % name)
    
    def delete_objects(self, name):
        self.objects[:] = [obj for obj in self.objects if obj.name != name]
    
    def direct_camera_towards(self, target):
        self.camera_direction = np.array(target)-np.array(self.camera_position)
        self.camera_sharp_distance = np.linalg.norm(self.camera_direction)
    
    def get_camera_rays(self):
        return utils.camera_rays(self.image_size, self.camera_flat_ccd, \
            self.camera_fov, self.camera_direction, self.camera_up)
    
    def get_number_of_camera_rays(self):
        return self.get_camera_rays().size / 3
    
    def add_object(self, tracer, material, name=None):
        self.objects.append(Object(tracer, material, name))

    def get_kernels(self, template_env):
        kernel_map = {}
        for obj in self.objects:
            for (k, v) in obj.tracer.make_functions(template_env).items():
                if k in kernel_map and kernel_map[k] != v:
                    print kernel_map[k]
                    print '------'
                    print v
                    raise RuntimeError("kernel name clash!!")
                kernel_map[k] = v
        return list(set(kernel_map.values()))
    
def default_materials():
    # --- Materials
    return {\
        'default': # "Air" / initial / default material
            { 'diffuse': ( 1, 0, 1),
              'emission': 0.0,
              'reflection': 0.0,
              'transparency': 0.0,
              'ior': 1.0, # Index Of Refraction
              'volume_absorption': 0.0,
              'volume_scattering': 0.0,
              'volume_scattering_blur': 1.0,
              'transparency_blur': 0.0,
              'reflection_blur': 0.0
            },
        # Other materials
        'white':
            { 'diffuse': 0.8 }, 
        'green':
            { 'diffuse': (0.4, 0.9, 0.4)},
        'red':
            { 'diffuse': (.7, .4, .4) }, 
        'mirror':
            { 'diffuse': 0.2, 'reflection': 0.7 },
        'light': # warm yellow-orange-light
            { 'diffuse': 1.0, 'emission':(4, 2, .7) },
        'sky':
            { 'diffuse': 0.0, 'emission':(.5, .5, .7) },
        'glass':
            { 'diffuse': 0.1, 'transparency': 0.7, 'reflection': 0.2, 'ior': 1.5 },
        'brushed metal':
            { 'diffuse': 0.2, 'reflection': 0.7, 'reflection_blur': 0.1 },
        'wax':
            { 'transparency': 0.9,
              'transparency_blur': 0.7,
              'volume_scattering': 0.5,
              'volume_scattering_blur': 1.0,
              'volume_absorption': (1.0-np.array([1,.8,.3])) * 4.0,
              'reflection': 0.05,
              'reflection_blur': 0.3,
              'diffuse': 0.02
            }
        }
    

# Default scene
class DefaultBoxScene(Scene):
    """
    Default raytracer scene. Override attributes as required
    """
    
    # helpers...
    
    @staticmethod
    def make_world_box( material, dims, center=(0, 0, 0) ):
        return [\
            Object(HalfSpace( ( 1, 0, 0), dims[0]-center[0] ), material, 'wall'), \
            Object(HalfSpace( (-1, 0, 0), dims[0]+center[0] ), material, 'wall'), \
            Object(HalfSpace( ( 0, 1, 0), dims[1]-center[1] ), material, 'wall'), \
            Object(HalfSpace( ( 0,-1, 0), dims[1]+center[1] ), material, 'wall'), \
            Object(HalfSpace( ( 0, 0, 1), dims[2]-center[2] ), material, 'floor'), \
            Object(HalfSpace( ( 0, 0,-1), dims[2]+center[2] ), material, 'ceiling')]
    
    
    def initialize_materials(self):
        self.materials = default_materials()
    
    def __init__(self):
        """Initialize default scene"""
        
        # --- Image settings
        self.image_size = (800, 600)
        self.brightness = 0.3
        self.gamma = 1.8
        
        # --- Raytracer settings
        self.tent_filter = True
        self.quasirandom = False
        self.samples_per_pixel = 10000
        self.min_bounces = 3
        self.russian_roulette_prob = .3
        self.max_bounces = 3
        
        self.initialize_materials()
        
        # --- Objects
        self.objects = DefaultBoxScene.make_world_box( 'white', (3, 5, 2), (0, 0, 2) )
        self.objects[-1].material = "sky" # world box ceiling
        self.objects[-2].material = "green" # world box floor
        self.root_object = None
        self.max_ray_length = 100
        
        # light bulb on the right wall
        self.objects.append(Object(Sphere( (-3, -1, 2), 0.5 ), 'light', 'light'))
        
        # --- Camera
        self.camera_up = (0, 0, 1)
        self.camera_position = (1, -5, 2)
        self.camera_fov = 55 # Field-of-view angle (horizontal)
        self.camera_flat_ccd = False
        camera_target = (0, 2, 0.5)
        self.camera_dof_fstop = 0.0
        self.direct_camera_towards(camera_target)
        
        self.shader = shader.RgbShader

class DefaultSpectrumBoxScene(DefaultBoxScene):
    
    def __init__(self):
        self.spectrum = Spectrum()
        DefaultBoxScene.__init__(self)
        self.shader = shader.SpectrumShader
    
    def initialize_materials(self):
        
        s = self.spectrum
        self.materials = default_materials()
        
        # Have to replace RGB colors by proper spectra...
        overrides = {
            'green':
                { 'diffuse': s.gaussian(540, 30)*0.65 + 0.3 },
            'red':
                { 'diffuse': s.gaussian(670, 30)*0.6 + 0.2 },
            'light': # warm yellow-orange-light
                { 'diffuse': 1.0, 'emission': s.black_body(3200)*7.0 },
            'sky':
                { 'diffuse': 1.0, 'emission': s.black_body(10000) },
        }
        
        for k, mat in overrides.items():
            self.materials[k] = mat
        
        self.materials['wax']['volume_absorption'] = \
            (1.0 - s.gaussian(670, 100)) * 4.0
            
        # Also make the glass dispersive
        self.materials['glass']['ior'] = s.linear_dispersion_ior(1.5, 60.0)
