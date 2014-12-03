
from scene import *
from utils import normalize, vec_norm

scene.materials['glass'] = {
	'diffuse': (.2,.2,.2),
	'transparency': (0.7,0.7,0.7),
	'reflection': (.1,.1,.1),
	'ior': (1.5) }

scene.materials['black'] = {
	'diffuse': (.2,.2,.2) }

scene.materials['blue'] = {
	'diffuse': (.1,.1,.2),
	'emission': (.001,.001,.3),
	'reflection': (0.7,0.7,0.8),
	'ior': (1.5)
}


scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }
	
scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

#fog = 0.8
#scene.materials['default']['vs'] = (fog,fog,fog)

for obj in scene.get_objects('wall'): obj.material = 'black'
scene.get_object('floor').material = 'green'
scene.get_object('ceiling').material = 'white'
scene.get_object('light').material = 'white'

#scene.get_objects('wall')[1].material = 'sky'

#obj = Cone( (0,0,1), (0,0,-1), 0.7, 0.8 )
#obj = Cone( (0,0,0), (0,0,1), 0.7, 0.8 )
#scene.objects.append( Object( obj, 'glass' ) )

#scene.objects.append( Object( Octahedron( (0,0,0.5), 0.5 ), 'white') )
scene.objects.append( Object( Sphere( (1.1,0.4,0.2), 0.2 ), 'light') )
scene.objects.append( Object( Icosahedron( (-0.4,1.0,0.3), 0.3 ), 'mirror') )
#scene.objects.append( Object( Tetrahedron( (0,0,0.5), 0.5 ), 'white') )

scene.objects.append( Object( Dodecahedron( (0.2,-0.5,0.3), 0.3 ), 'white') )

#scene.image_size = (1024,768)

scene.samples_per_pixel = 600000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((-0.1,0,0.2))
scene.camera_fov = 40
#scene.gamma =  1.5
#scene.brightness = 0.25

scene.min_bounces = 2
scene.max_bounces = 2

#self.camera_position = (1,-5,2)
#scene.camera_dof_fstop = 0.1
scene.camera_sharp_distance = vec_norm(scene.camera_position)
