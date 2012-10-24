
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

#scene.get_object('ceiling').material = 'red' 
#scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

import sympy
tanglecube_eq = 'x^4 - 5*x^2 + y^4 - 5*y^2 + z^4 - 5*z^2 + 11.8'
tanglecube_eq = sympy.sympify(tanglecube_eq)
#tanglecube_eq += 1
K = '5';
A = '0.95';
B = '0.8';
chair_eq = '((x^2 + y^2 + z^2 - a*k^2)^2 - b*( (z-k)^2 - 2*x^2 ) * ( (z+k)^2 - 2*y^2))'
chair_eq = chair_eq.replace('k', K).replace('a', A).replace('b', B)

#eq = sympy.sympify(chair_eq) + tanglecube_eq*0.03
#impsurf = ImplicitSurface(eq, center=(0,0,1), scale=0.2, bndR=6, max_itr=1000, precision=0.001)

#eq = sympy.sympify(chair_eq)
#impsurf = ImplicitSurface(eq, center=(0,0,1), scale=0.2, bndR=6, max_itr=1000, precision=0.0001)

#eq = tanglecube_eq
#impsurf = ImplicitSurface(eq, center=(0,0,0.5), scale=0.25, bndR=4, max_itr=1500, precision=0.001)

impsurf = QuaternionJuliaSet2(
	c=(-0.5, 0.4, -0.5, -0.1), julia_itr=7,
	center=(0,0,1), scale=1, bndR = 1.5,
	max_itr=2000, precision=0.0001)

scene.objects.append( Object( impsurf, 'white' ) )

scene.image_size = (640,400)
#scene.image_size = (320,200)

scene.samples_per_pixel = 10000
scene.camera_position = (2,3,3.5)
scene.direct_camera_towards((0,0,1))
scene.camera_fov = 60

scene.min_bounces = scene.max_bounces = 2

#scene.min_bounces = 4
#scene.max_bounces = 6
