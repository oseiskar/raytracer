
from scene import *
from utils import normalize, vec_norm

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

#obj = Cone( (0,0,1), (0,0,-1), 0.7, 0.8 )
#obj = Cone( (0,0,0), (0,0,1), 0.7, 0.8 )
#scene.objects.append( Object( obj, 'glass' ) )

"""
scene.objects.append( Object(
	ConvexIntersection( (0,0,0.5), [
		ConeComponent( (0,0,-0.5), (0,0,1), 1.0 ),
		ConeComponent( (0,0,0.5), (0,0,-1), 1.0 )
	] ), 'glass') )
"""
#scene.objects.append( Object( Octahedron( (0,0,0.5), 0.5 ), 'white') )
#scene.objects.append( Object( Dodecahedron( (0,0,0.5), 0.5 ), 'white') )
#scene.objects.append( Object( Icosahedron( (0,0,0.5), 0.5 ), 'white') )
scene.objects.append( Object( Tetrahedron( (0,0,0.5), 0.5 ), 'white') )

scene.image_size = (800,600)

scene.samples_per_pixel = 1000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 40

scene.min_bounces = 2
scene.max_bounces = 3

#self.camera_position = (1,-5,2)
#scene.camera_dof_fstop = 0.1
scene.camera_sharp_distance = vec_norm(scene.camera_position)
