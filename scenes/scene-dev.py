
from scene import *
from utils import normalize

scene.materials['glass'] = {
	'diffuse': (.3,.3,.3),
	#'diffuse': (.0,.0,.0),
	'transparency': (.6,.6,.6),
	'reflection': (.1,.1,.1),
	'ior': (1.5,) }

scene.materials['black'] = {
	'diffuse': (.2,.2,.2) }

scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }

for obj in scene.get_objects('wall'): obj.material = 'black'
scene.get_object('floor').material = 'green'

#scene.get_object('ceiling').material = 'red' 
#scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

#obj = Cylinder( (0,0,0.0), (0,2,1), 0.7, 0.8 )
#obj = Parallelepiped( (0,0,0), (.7,.0,.3), (0,.3,0), (0,0,.8) )
#obj = Sphere( (0,0,.5), 0.5 )
obj = Cone( (0,0,0.0), (0,0,1), 0.7, 0.8 )
scene.objects.append( Object( obj, 'glass' ) )

#scene.objects.append( Object( Sphere((0,0,0.7), 0.3) , 'black' ) )
#scene.objects.append( Object( Cylinder((0,0,0.65), (0,0,1), 0.1, 0.3) , 'mirror' ) )



#scene.image_size = (1024,768)
scene.image_size = (800,600)
#scene.image_size = (640,400)
#scene.image_size = (320,200)

scene.samples_per_pixel = 100000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 40

#scene.min_bounces = scene.max_bounces = 2
#scene.min_bounces = 4
scene.max_bounces = 6
