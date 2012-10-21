
from scene import *
from utils import normalize

scene.materials['glass'] = {
	'diffuse': (.1,.1,.1),
	'transparency': (.35,.35,.6),
	'reflection': (.2,.2,.3),
	'ior': (1.5,) }

scene.materials['sky'] = {
	'diffuse': ( 0, 0, 0), 'emission': tuple(np.array((.5,.5,.7))*0.7) }

scene.get_object('floor').material = 'red' # red floor
scene.get_object('ceiling').material = 'white'  # white ceiling

scene.objects.append( Object(HalfSpace( tuple(normalize(np.array((-1,-1,-2)))), 5 ), 'sky') )

scene.objects.append( Object(Sphere( (-0.2,2.5,1.2), 1.2 ), 'green') )
scene.objects.append( Object(Sphere( (-0.7,-0.8,.4), .4 ), 'glass') )
tanglecube_eq = 'x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8'
scene.objects.append( Object( ImplicitSurface(tanglecube_eq, (1.8,0.2,0.5), 0.25, 4), 'mirror' ) )

scene.image_size = (640,400)

