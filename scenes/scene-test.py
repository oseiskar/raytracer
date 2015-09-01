"""
Test scene: should contain all different objects
"""

from scene import *
import math
import numpy
import mesh_formats
from transformations import Affine

def load_triangle_mesh(p, R, **kwargs):
    from StringIO import StringIO
    
    mesh_data = "\n".join([
        'OFF',
        '8 6 24',
        '0 0 0',
        '0 0 1',
        '0 1 0',
        '0 1 1',
        '1 0 0',
        '1 0 1',
        '1 1 0',
        '1 1 1',
        '4 0 1 3 2',
        '4 2 3 7 6',
        '4 4 6 7 5',
        '4 0 4 5 1',
        '4 1 5 7 3',
        '4 0 2 6 4'
    ])
    
    vertices, faces = mesh_formats.read_off(StringIO(mesh_data))
    faces = mesh_formats.remove_duplicate_faces(faces)
    triangle_mesh = TriangleMesh(vertices, faces, \
        auto_scale=True, center=p, scale=R, **kwargs)
    return triangle_mesh

def load_triangle_mesh_in_octree(p, R, **kwargs):
    return Octree(load_triangle_mesh(p,R, **kwargs), max_depth=2, max_faces_per_leaf=1)

test_objects = [
    lambda p, R: Sphere(p,R),
    lambda p, R: Cone(p-numpy.array([0,0,R]), (0,0,1), R, R ),
    lambda p, R: Cylinder(p-numpy.array([0,0,R]), (0,0,1), R, R ),
    lambda p, R: Tetrahedron( p, R ),
    lambda p, R: Octahedron( p, R ),
    lambda p, R: Dodecahedron( p, R ),
    lambda p, R: Icosahedron( p, R ),
    lambda p, R: Parallelepiped( p - numpy.array((R*0.5,R*0.45,R)), (R,0,0), (0,R*0.9,0), (0,0,1.5*R) ),
    lambda p, R: QuaternionJuliaSet( (-0.2,-0.4,-0.4,-0.4), 4, center=p, scale=R ),
    lambda p, R: QuaternionJuliaSet2( (-0.2,-0.4,-0.4,-0.4), 4, center=p, scale=R ),
    lambda p, R: ImplicitSurface('x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8', p, R*0.5, 4),
    lambda p, R: ConvexIntersection( p, [ \
        CylinderComponent( (0,1,0), R, ), \
        ConeComponent( (0,-R/0.2,0), (0,1,0), R, ), \
        SphereComponent( (0,0,0), R*1.1, ),
        HalfSpaceComponent( (1,1,0), R ),
        LayerComponent( (1,0,0), 0.3*R/0.2 ) ]),
    load_triangle_mesh,
    lambda p,R: load_triangle_mesh(p,R, auto_flip_normal=True),
    lambda p,R: load_triangle_mesh(p,R, shading='smooth', auto_smooth_normals=True),
    # another Octahedron (should have same tracer but different size)
    lambda p, R: Octahedron( p, R*0.5 ),
    # another ConvexIntersection, should have different tracer
    lambda p, R: ConvexIntersection( p, [ \
        CylinderComponent( (1,1,0), R, ), \
        SphereComponent( (0,0,0), R*1.2, ),
        LayerComponent( (1,0,0), 2.0*R ) ]),
    # Octree
    load_triangle_mesh_in_octree,
    lambda p, R: DistanceField( tracer_code="dist = sqrt(x*x + y*y + z*z) - %g" % R, center=p )
]

test_materials = [
    'white', 'mirror', 'glass', 'wax'
]

scene = DefaultBoxScene()
scene.get_object('light').bidirectional_light = True

grid_side = int(math.ceil(math.sqrt(len(test_objects))))
grid = [(x,y) for x in range(grid_side) for y in range(grid_side)]
grid_size = 4.0
scale = 0.5*grid_size/grid_side
index_to_coord = lambda ix: ((ix+0.5)/float(grid_side)-0.5)*grid_size

for i in range(len(test_objects)):
    ix, iy = grid[i]
    material = test_materials[i % len(test_materials)]
    obj_scale = 0.5
    z = scale*obj_scale
    x = index_to_coord(ix)
    y = index_to_coord(iy)
    pos = numpy.array((x,y,z))
    tracer = test_objects[i](pos,scale*obj_scale)
    #tracer = test_objects[i]((0,0,0),1.0)
    tracer.rotate(axis='z', deg=-45)
    scene.objects.append( Object( tracer, material ) )

scene.max_bounces = 4
scene.min_bounces = 2

scene.direct_camera_towards((0,0,0.0))
