from scene import *
from utils import normalize, vec_norm

#scene = DefaultSpectrumBoxScene()
scene = DefaultBoxScene()

for obj in scene.get_objects('wall'): obj.material = 'red'
scene.get_object('floor').material = 'white'
scene.get_object('ceiling').material = 'white'
#scene.get_object('light').material = 'white'
scene.get_object('light').bidirectional_light = True
scene.materials['black'] = { 'diffuse': 0.2 }

import mesh_formats

objR = .6
objPos = (0,0,objR)
objMat = 'white'
#objType = Tetrahedron
#objType = Octahedron
#objType = Dodecahedron
#objType = Icosahedron
#objType = Sphere
#scene.objects.append( Object( objType( objPos, objR ), objMat ) )

vertices,faces = mesh_formats.read_zipper('data/bun_zipper.ply')
faces = mesh_formats.remove_duplicate_faces(faces)
#vertices,faces = mesh_formats.read_off('data/S.off')
vertices = [[x,-z,y] for x,y,z in vertices]

print "%d vertices, %d faces" % (len(vertices),len(faces))

#print faces
obj = TriangleMesh( vertices, faces, center=objPos, scale=objR, auto_scale=True, auto_normal=False )
do_octree = True

if do_octree:
    print 'computing octree'
    octree = Octree(obj, max_depth=4, max_faces_per_leaf=10)
    print 'done'
    scene.objects.append( Object(octree, objMat) )
else:
    scene.objects.append( Object( obj, objMat ) )

szmul = 120
scene.image_size = (8*szmul,6*szmul)

scene.samples_per_pixel = 15000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 1
scene.max_bounces = 1
