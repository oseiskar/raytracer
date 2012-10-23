
from scene import *
from utils import normalize

scene.materials['glass'] = {
	'diffuse': (.3,.3,.3),
	'transparency': (.6,.6,.6),
	'reflection': (.1,.1,.1),
	'ior': (1.5,) }

scene.materials['black'] = {
	'diffuse': (.2,.2,.2) }

scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }

for obj in scene.get_objects('wall'): obj.material = 'black'
scene.get_object('floor').material = 'green'
scene.get_object('ceiling').material = 'red' 

scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

tanglecube_eq = 'x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8'
scene.objects.append( Object( ImplicitSurface(tanglecube_eq, (0,0,.7), 0.7/2, 4), 'glass', 'tanglecube' ) )

#scene.delete_objects('light')
#scene.get_object('light').tracer.pos = (0,0,0.7)
#scene.get_object('light').tracer.R = 0.2

scene.image_size = (800,600)

scene.samples_per_pixel = 10000
scene.camera_position = (2,-3,3.5)
scene.direct_camera_towards((0,0,0.5))
#scene.camera_position = ((0,0,1))
#scene.camera_direction = ((1,1,1))
scene.camera_fov = 55
scene.min_bounces = 4
scene.max_bounces = 6
