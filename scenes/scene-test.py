"""
Test scene: should contain all different objects
"""

from scene import *
import math
import numpy

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
    lambda p, R: ImplicitSurface('x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8', p, R*0.5, 4),
    lambda p, R: ConvexIntersection( p, [ \
        CylinderComponent( (0,1,0), R, ), \
        ConeComponent( (0,-1,0), (0,1,0), R, ), \
        SphereComponent( (0,0,0), R*1.1, ),
        HalfSpaceComponent( (1,1,0), 0.2 ),
        LayerComponent( (1,0,0), 0.3 ) ]),
    # TODO: does not have a scale argument
    #lambda p, R: QuaternionJuliaSet2( (-0.2,-0.4,-0.4,-0.4), 4, center=p, scale=R )
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
    scene.objects.append( Object( test_objects[i](numpy.array((x,y,z)),scale*obj_scale), material ) )

scene.max_bounces = 3
scene.min_bounces = 3

scene.direct_camera_towards((0,0,0.0))
