from scenes.default_scenes import BoxScene
from objects import *
#from shader import SpectrumShader

#scene = BoxScene(SpectrumShader)
scene = BoxScene()

for obj in scene.get_objects('wall'): obj.material = 'red'
scene.get_object('floor').material = 'white'
scene.get_object('ceiling').material = 'green'
#scene.get_object('light').material = 'white'
scene.get_object('light').bidirectional_light = True
scene.materials['black'] = { 'diffuse': 0.2 }
scene.materials['light']['diffuse'] = 0.0
scene.add_object( HalfSpace( (-1,-1,-2), 5 ), 'sky' )

obj = Sphere((0,0,0.6), 0.3)
obj.linear_transform(scaling=(1,1,2))
scene.add_object( obj, 'mirror' )

szmul = 120
scene.image_size = (8*szmul,6*szmul)

scene.samples_per_pixel = 15000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 2
scene.max_bounces = 5
