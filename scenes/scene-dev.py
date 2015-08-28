from scene import *
from utils import normalize, vec_norm

scene = DefaultSpectrumBoxScene()
#scene = DefaultBoxScene()

for obj in scene.get_objects('wall'): obj.material = 'red'
scene.get_object('floor').material = 'white'
scene.get_object('ceiling').material = 'green'
#scene.get_object('light').material = 'white'
scene.get_object('light').bidirectional_light = True
scene.materials['black'] = { 'diffuse': 0.2 }
scene.materials['light']['diffuse'] = 0.0
scene.objects.append( Object(HalfSpace( (-1,-1,-2), 5 ), 'sky') )

objR = .6
objPos = (0,0,objR)
objMat = 'white'

distance_field = """
    float th = z*z * 4.0;
    float x1 = x * cos(th) - y * sin(th);
    float y1 = x * sin(th) + y * cos(th);
    
    dist = max(fabs(x1) - 0.3, fabs(y1) - 0.4);
    dist = max(dist, z - 0.85);
"""
obj = DistanceField( tracer_code=distance_field, center=objPos, self_intersection=False )

scene.objects.append( Object( obj, objMat ) )

szmul = 120
scene.image_size = (8*szmul,6*szmul)

scene.samples_per_pixel = 15000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 2
scene.max_bounces = 5
