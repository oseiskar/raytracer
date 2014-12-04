from scene import *
from utils import normalize, vec_norm

scene.materials['glass'] = {
	'diffuse': (.15,.15,.15),
	'diffuse': (.0,.0,.0),
	'transparency': (.7,.7,.7),
	'reflection': (.1,.1,.1),
	'ior': (1.5,),
    'dispersion': (0.1,) }

scene.materials['black'] = {
	'diffuse': (.2,.2,.2) }

scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }

for obj in scene.get_objects('wall'): obj.material = 'black'
scene.get_object('floor').material = 'green'
scene.get_object('ceiling').material = 'white'

scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

objR = .6
objPos = (0,0,objR)
objMat = 'glass'
#objType = Tetrahedron
#objType = Octahedron
#objType = Dodecahedron
#objType = Icosahedron
objType = Sphere

scene.objects.append( Object( objType( objPos, objR ), objMat ) )

"""
scene.objects.append( Object(
	ConvexIntersection( (-0,0,objR), [
		CylinderComponent( (1,0,0), objR, ),
		CylinderComponent( (0,1,0), objR, ),
		CylinderComponent( (0,0,1), objR, )
	] ), 'glass') )
"""

scene.image_size = (800,600)

scene.samples_per_pixel = 5000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 2
scene.max_bounces = 4

#self.camera_position = (1,-5,2)
#scene.camera_dof_fstop = 0.1
scene.camera_sharp_distance = vec_norm(scene.camera_position)
