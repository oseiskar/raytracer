
from scene import *
from utils import normalize
import math

scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }

scene.get_object('floor').material = 'red' # red floor
scene.get_object('ceiling').material = 'white'  # white ceiling

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

scene.objects.append( Object( Parallelepiped( (1.3,-0.5,0.0), (1,0,0), (0,1.3,0), (0,0,0.6) ), 'mirror' ) )

scene.objects.append( Object(
	ConvexIntersection( (1.8,0.2,.5+0.6), [
		SphereComponent( (0,-.5,0), math.sqrt(2)*.5, ),
		SphereComponent( (0,.5,0), math.sqrt(2)*.5, )
	] ), 'glass') )

#scene.image_size = (1024,768)
scene.image_size = (640,400)
scene.samples_per_pixel = 1000 #80000
scene.quasirandom = False
scene.gamma = 1.6
#scene.max_bounces = 5
