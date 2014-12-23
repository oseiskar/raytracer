from scene import *
from utils import normalize, vec_norm

#scene = DefaultSpectrumBoxScene()
scene = DefaultBoxScene()
#spectrum = scene.spectrum

for obj in scene.get_objects('wall'): obj.material = 'red'
scene.get_object('floor').material = 'white'
scene.get_object('ceiling').material = 'white'
scene.get_object('light').material = 'white'
scene.materials['black'] = { 'diffuse': 0.2 }
#scene.materials['glass']['ior'] = spectrum.linear_dispersion_ior(1.6, 36.0)
#scene.materials['sky']['emission'] = spectrum.black_body(5000)

scene.materials['default']['volume_scattering'] = 0.1
scene.materials['default']['volume_scattering_blur'] = 0.01

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

szmul = 50
scene.image_size = (8*szmul,6*szmul)

#tanglecube_eq = 'x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8'
#scene.objects.append( Object( ImplicitSurface(tanglecube_eq, (0,0,0.5), 0.25, 4), 'wax' ) )

scene.samples_per_pixel = 15000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 3
scene.max_bounces = 3
