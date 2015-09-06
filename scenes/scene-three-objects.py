
from objects import *
from utils import vec_norm
from scenes.default_scenes import BoxScene
import numpy as np

scene = BoxScene()
	
scene.materials['floor'] = {
	'diffuse': 0.1,
	'reflection': 0.15,
    'reflection_blur': 0.05
}

scene.materials['glass']['transparency'] = (0.4, 0.6, 0.9)
scene.materials['glass']['ior'] = 1.5

scene.materials['wax']['volume_absorption'] = (1.0 - np.array((0.2,0.8,0.4)))*7.0
scene.materials['wax']['diffuse'] = (0.02,0.03,0.04)
scene.materials['wax']['reflection'] = (0.01,0.04,0.02)

scene.get_object('floor').material = 'floor'
scene.get_object('ceiling').material = 'white'
scene.get_object('light').bidirectional_light = True

for obj in scene.get_objects('wall'): obj.material = 'white'

#scene.materials['default']['volume_scattering'] = 0.15
#scene.materials['default']['volume_scattering_blur'] = 0.03

scene.add_object(HalfSpace( (-1,-1,-2), 5 ), 'sky')

scene.add_object(
	ConvexIntersection( (-0.2,2.5,1.0), [
		CylinderComponent( (1,1,0), 1, ),
		CylinderComponent( (0,1,0), 1, ),
		CylinderComponent( (1,0,1), 1, )
	] ), 'mirror')
	
cylR = .4
scene.add_object(
	ConvexIntersection( (-0.7,-0.8,cylR), [
		CylinderComponent( (1,0,0), cylR, ),
		CylinderComponent( (0,1,0), cylR, ),
		CylinderComponent( (0,0,1), cylR, )
	] ), 'wax')

scene.add_object(Parallelepiped( (1.3,-0.5,0.0), (1,0,0), (0,1.3,0), (0,0,0.6) ), 'red')

scene.add_object(Dodecahedron( (1.8,0.2,.5+0.6), 0.5 ), 'glass')

#scene.image_size = (1280,1024)
scene.image_size = (1024,768)
#scene.image_size = (640,400)
scene.samples_per_pixel = 10000
scene.quasirandom = False
scene.gamma = 1.6
scene.max_bounces = 5
scene.camera_fov = 60

scene.camera_position = (-2,-3,1.5)
scene.direct_camera_towards( (0,0,0.5) )
#scene.camera_dof_fstop = 0.02
scene.camera_sharp_distance = vec_norm( np.array([1.8,0.2,.5+0.6]) - np.array(scene.camera_position) )
