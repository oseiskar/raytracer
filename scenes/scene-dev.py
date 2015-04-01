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

def poly_to_triangle_fan(vertices):
    for offset in range(len(vertices)-2):
        yield([vertices[0],vertices[offset+1],vertices[offset+2]])

def read_off(filename):
    with open(filename, 'r') as off:
        
        first_line = off.readline().strip()
        none, sep, rest = first_line.partition('OFF')
        if len(none) > 0 or sep != 'OFF':
            raise RuntimeError('not OFF format')
        rest = rest.strip()
        if len(rest) == 0:
            rest = off.readline().strip()
        n_points, n_faces, _ = [int(x) for x in rest.split()]
        
        print 'reading OFF with', n_points, 'points and', n_faces, 'faces'
        
        vertices = []
        for v in xrange(n_points):
            vertices.append([float(x) for x in off.readline().split()])
        
        faces = []
        for f in xrange(n_faces):
            face_vertices = [int(x) for x in off.readline().split()]
            if face_vertices[0] != len(face_vertices)-1:
                raise "invalid format"
            
            faces += list(poly_to_triangle_fan(face_vertices[1:]))
            
        print "constructed", len(faces), "triangles"
        return vertices, faces

objR = .6
objPos = (0,0,objR)
objMat = 'white'
#objType = Tetrahedron
#objType = Octahedron
#objType = Dodecahedron
#objType = Icosahedron
#objType = Sphere
#scene.objects.append( Object( objType( objPos, objR ), objMat ) )

vertices,faces = read_off('data/S.off')
#print faces
obj = TriangleMesh( vertices, faces, center=objPos, scale=objR, auto_scale=True, auto_normal=False )
print 'computing octree'
octree = Octree(obj)
print 'done'
#scene.objects.append( Object( obj, objMat ) )
scene.objects.append( Object(octree, objMat) )

dat = octree.get_data()
#print dat
#print octree.total_faces
#print octree.root_data_offset
print dat['integer'][octree.total_faces*3:]
print dat['integer'][octree.total_faces*3+octree.root_data_offset:]
#print len(octree.leafs)

#for node in octree.leafs:
#    if node.is_empty(): continue
#    obj = Sphere( *node.get_bounding_sphere() )
#    scene.objects.append( Object( obj, objMat ) )

szmul = 120
scene.image_size = (8*szmul,6*szmul)

scene.samples_per_pixel = 15000
scene.camera_position = (-2,-3,1)
scene.direct_camera_towards((0,0,0.6))
scene.camera_fov = 50

scene.min_bounces = 1
scene.max_bounces = 1
