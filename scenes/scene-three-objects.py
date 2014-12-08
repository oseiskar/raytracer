
from scene import *
from utils import normalize, vec_norm
import math

#scene = DefaultSpectrumBoxScene()
scene = DefaultBoxScene()
	
scene.materials['floor'] = {
	'diffuse': ( 0.2, 0.2, 0.2),
	'reflection':(.2,.2,.2)
}

scene.get_object('floor').material = 'floor'
scene.get_object('ceiling').material = 'white'

scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

scene.objects.append( Object(
	ConvexIntersection( (-0.2,2.5,1.0), [
		CylinderComponent( (1,1,0), 1, ),
		CylinderComponent( (0,1,0), 1, ),
		CylinderComponent( (1,0,1), 1, )
	] ), 'mirror') )
	
cylR = .4
scene.objects.append( Object(
	ConvexIntersection( (-0.7,-0.8,cylR), [
		CylinderComponent( (1,0,0), cylR, ),
		CylinderComponent( (0,1,0), cylR, ),
		CylinderComponent( (0,0,1), cylR, )
	] ), 'green') )

scene.objects.append( Object( Parallelepiped( (1.3,-0.5,0.0), (1,0,0), (0,1.3,0), (0,0,0.6) ), 'red' ) )

scene.objects.append( Object(
	ConvexIntersection( (1.8,0.2,.5+0.6), [
		SphereComponent( (0,-.5,0), math.sqrt(2)*.5, ),
		SphereComponent( (0,.5,0), math.sqrt(2)*.5, )
	] ), 'glass') )

#scene.image_size = (1280,1024)
scene.image_size = (1024,768)
#scene.image_size = (640,400)
scene.samples_per_pixel = 10000
scene.quasirandom = False
scene.gamma = 1.6
#scene.max_bounces = 5
scene.camera_fov = 60

scene.camera_position = (-2,-3,1.5)
scene.direct_camera_towards( (0,0,0.5) )
scene.camera_dof_fstop = 0.05
scene.camera_sharp_distance = vec_norm( np.array([1.8,0.2,.5+0.6]) - np.array(scene.camera_position) )
