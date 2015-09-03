from objects import ConvexIntersection, HalfSpaceComponent, LayerComponent, FixedConvexIntersection
from utils import vec_norm
import math

class Parallelepiped(ConvexIntersection):
    
    def __init__(self, origin, ax1, ax2, ax3):
        components = [ LayerComponent(ax) for ax in (ax1, ax2, ax3)]
        ConvexIntersection.__init__(self, origin, components)
        self.unique_tracer_id = ''

class HyperplaneRepresentation(FixedConvexIntersection):
    """Hyperplane representation of a polyhedron"""
    
    def __init__( self, center, planes, R ):
        # TODO: rather add R as an attribute to Tracer...
        components = []
        for p in planes:
            halfspace = HalfSpaceComponent( p, vec_norm(p) )
            components.append(halfspace)
        
        FixedConvexIntersection.__init__(self, center, components)
        self.linear_transform(scaling=R)
    

class SymmetricDualPolyhedron(FixedConvexIntersection):
    
    def __init__( self, center, vertices, R ):
        """
        The vertices and their symmetries about the center (i.e.,
        v -> -v) define the dual polyhedron of the result. R is a
        scaling factor. If R is 1 and the center is (0,0,1), then 
        a vertex, face or edge of the polyhedron is on the ground
        plane {z = 0}.
        """
        
        components = []
        for v in vertices:
            layer = LayerComponent( tuple([-x for x in v]), vec_norm(v)*2.0 )
            layer.position = v
            components.append(layer)
        
        FixedConvexIntersection.__init__(self, center, components)
        self.linear_transform(scaling=R)

class Tetrahedron(HyperplaneRepresentation):
    def __init__(self, center, R):
        rsqrt2 = 1.0/ math.sqrt(2)
        HyperplaneRepresentation.__init__(self, center,
            [(1,0,-rsqrt2), # vertices of a tetrahedron
             (-1,0,-rsqrt2),
             (0,1,rsqrt2),
             (0,-1,rsqrt2)],
            R / (rsqrt2 + 1/rsqrt2))

class Cube(SymmetricDualPolyhedron):
    def __init__(self, center, side=None, R=None):
        if side is None:
            if R is None: side = 1.0
            side = R/math.sqrt(3)*2.0
        SymmetricDualPolyhedron.__init__(self, center,
            [(1,0,0),
             (0,1,0),
             (0,0,1)],
            side / 2.0)

class Octahedron(SymmetricDualPolyhedron):
    def __init__(self, center, R):
        SymmetricDualPolyhedron.__init__(self, center,
            [(1,1,1), # vertices of a cube
             (1,-1,1),
             (-1,1,1),
             (-1,-1,1)],
            R / 3.0)

class Dodecahedron(SymmetricDualPolyhedron):
    def __init__(self, center, R):
        phi = 0.5 * (1 + math.sqrt(5)) # the golden ratio
        SymmetricDualPolyhedron.__init__(self, center,
            [(0,1,phi), # vertices of an icosahedron
             (1,phi,0),
             (phi,0,1),
             (0,-1,phi),
             (-1,phi,0),
             (phi,0,-1)],
            R / (phi + 1.0/phi))

class Icosahedron(SymmetricDualPolyhedron):
    def __init__(self, center, R):
        phi = 0.5 * (1 + math.sqrt(5)) # the golden ratio
        SymmetricDualPolyhedron.__init__(self, center,
            [(1,1,1), # vertices of a dodecahedron
             (-1,1,1),
             (1,-1,1),
             (-1,-1,1),
             (0,1/phi,phi),
             (0,-1/phi,phi),
             (1/phi,phi,0),
             (-1/phi,phi,0),
             (phi,0,1/phi),
             (phi,0,-1/phi)],
            R / (phi + 1/phi**3))
