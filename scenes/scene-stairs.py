from objects import Sphere, Cube, HalfSpace
from scenes.default_scenes import DefaultScene
from shader import SpectrumShader
import numpy as np

from transformations import Affine

class DomeScene(DefaultScene):
    
    def set_up_objects_and_camera(self):
        
        self.materials['dome'] = {
            'diffuse': 0.0,
        }
        self.materials['red'] = {
            'diffuse': self.spectrum.gaussian(670, 30)*0.9 + 0.05,
            'reflection':  0.1,
            'reflection_blur': 0.05
        }
        
        sky_distance = 100
        dome = self.add_object(Sphere((0, 0, 0), sky_distance), 'dome', 'dome')
        self.add_object(HalfSpace((0, 0, -1), sky_distance*0.7 ), 'sky', 'sky')
        self.add_object(HalfSpace( (0, 0, 1), 0.0 ), 'white', 'floor')
        self.root_object = dome
        
        self.camera_position = (1.5, -7.5, 1.5)
        camera_target = (-0.5, 0, 1.5)
        self.direct_camera_towards(camera_target)

scene = DomeScene(SpectrumShader)
scene.camera_dof_fstop = 0.05

scene.image_size = (1920/2, 1080/2)
scene.samples_per_pixel = 100000

T = Affine(translation=(1,0,0))
dt = Affine(rotation_axis=(-0.5,-1,3.1), rotation_deg=10, translation=(0,0,0.07))

T2 = Affine(translation=(0,1.5,0))

for j in range(65):
    cube = Cube((0,0,0), 1.0)
    
    if j % 16 == 8:
        cube_mat = 'light'
    elif j % 16 == 0:
        cube_mat = 'red'
    else:
        cube_mat = 'glass'
        
    cube.coordinates = T * Affine(scaling=(0.5, 0.2, 0.02))
    scene.add_object(cube, cube_mat)
    
    sphere = Sphere((0,0,0), 0.1)
    sphere.coordinates = T2
    scene.add_object(sphere, 'red')
    
    T = dt * T
    T2 = dt * T2
