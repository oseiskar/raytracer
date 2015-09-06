
import numpy as np
from scene import Scene
from spectrum import Spectrum
from objects import HalfSpace, Sphere
import shader

def default_rgb_materials():
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

def default_spectrum_materials(spectrum):
    
    materials = default_rgb_materials()
        
    # Have to replace RGB colors by proper spectra...
    overrides = {
        'green':
            { 'diffuse': spectrum.gaussian(540, 30)*0.65 + 0.3 },
        'red':
            { 'diffuse': spectrum.gaussian(670, 30)*0.6 + 0.2 },
        'light': # warm yellow-orange-light
            { 'diffuse': 1.0, 'emission': spectrum.black_body(3200)*7.0 },
        'sky':
            { 'diffuse': 1.0, 'emission': spectrum.black_body(10000) },
    }
    
    for k, mat in overrides.items():
        materials[k] = mat

    materials['wax']['volume_absorption'] = \
        (1.0 - spectrum.gaussian(670, 100)) * 4.0
        
    # Also make the glass dispersive
    materials['glass']['ior'] = spectrum.linear_dispersion_ior(1.5, 60.0)
    
    return materials

def default_settings(scene):

    # --- Image settings
    scene.image_size = (800, 600)
    scene.brightness = 0.3
    scene.gamma = 1.8
    scene.brightness_reference = 'mean'
    
    # --- Raytracer settings
    scene.tent_filter = True
    scene.quasirandom = False
    scene.samples_per_pixel = 10000
    scene.min_bounces = 2
    scene.max_bounces = 4
    scene.min_russian_prob = 0.15
    
    # --- Default camera settings
    scene.camera_up = (0, 0, 1)
    scene.camera_fov = 55 # Field-of-view angle (horizontal)
    scene.camera_flat_ccd = False
    scene.camera_dof_fstop = 0.0
    
    scene.root_object = None
    scene.max_ray_length = 1000

def make_world_box(scene, material, dims, center=(0, 0, 0) ):
    
    for (obj, name) in [\
            (HalfSpace( ( 1, 0, 0), dims[0]-center[0] ), 'wall'), \
            (HalfSpace( (-1, 0, 0), dims[0]+center[0] ), 'wall'), \
            (HalfSpace( ( 0, 1, 0), dims[1]-center[1] ), 'wall'), \
            (HalfSpace( ( 0,-1, 0), dims[1]+center[1] ), 'wall'), \
            (HalfSpace( ( 0, 0, 1), dims[2]-center[2] ), 'floor'), \
            (HalfSpace( ( 0, 0,-1), dims[2]+center[2] ), 'ceiling')]:
        scene.add_object(obj, material, name)

class DefaultScene(Scene):
    
    def __init__(self, shader_class = shader.RgbShader):
        default_settings(self)
        self.shader = shader_class
        
        if shader_class == shader.SpectrumShader:
            self.spectrum = Spectrum()
            self.materials = default_spectrum_materials(self.spectrum)
        else:
            self.materials = default_rgb_materials()
        
        self.objects = []
        self.set_up_objects_and_camera()

class BoxScene(DefaultScene):
    """
    Default raytracer scene. Override attributes as required
    """
    
    def set_up_objects_and_camera(self):
        
        # --- Objects
        make_world_box(self, 'white', (3, 5, 2), (0, 0, 2))
        self.objects[-1].material = "sky" # world box ceiling
        self.objects[-2].material = "green" # world box floor
        
        # light bulb on the right wall
        self.add_object(Sphere((-3, -1, 2), 0.5), 'light', name='light')
        
        # --- Camera
        self.camera_position = (1, -5, 2)
        self.camera_fov = 55 # Field-of-view angle (horizontal)
        camera_target = (0, 2, 0.5)
        self.direct_camera_towards(camera_target)

