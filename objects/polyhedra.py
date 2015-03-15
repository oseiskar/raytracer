from objects import ConvexIntersection, HalfSpaceComponent, LayerComponent
import math, numpy
from utils import vec_norm, normalize_tuple

class Parallelepiped(ConvexIntersection):
	
	def __init__(self, origin, ax1, ax2, ax3):
		components = [ LayerComponent(ax) for ax in (ax1,ax2,ax3)]
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = ''

class HyperplaneRepresentation(ConvexIntersection):
	"""Hyperplane representation of a polyhedron"""
	
	def __init__( self, origin, planes, R ):
		# TODO: rather add R as an attribute to Tracer...
		components = []
		for unscaledP in planes:
			p = tuple([R*x for x in unscaledP])
			halfspace = HalfSpaceComponent( p, vec_norm(p) )
			components.append(halfspace)
		
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = str(R).replace('.','_')

class SymmetricDualPolyhedron(ConvexIntersection):
	
	def __init__( self, origin, vertices, R ):
		"""
		The vertices and their symmetries about the origin (i.e.,
		v -> -v) define the dual polyhedron of the result. R is a
		scaling factor. If R is 1 and the origin is (0,0,1), then 
		a vertex, face or edge of the polyhedron is on the ground
		plane {z = 0}.
		"""
		
		components = []
		for unscaledV in vertices:
			v = tuple([R*x for x in unscaledV])
			layer = LayerComponent( tuple([-x for x in v]), vec_norm(v)*2.0 )
			layer.pos = v
			components.append(layer)
		
		ConvexIntersection.__init__(self, origin, components)
		self.unique_tracer_id = str(R).replace('.','_')



class Tetrahedron(HyperplaneRepresentation):
	def __init__(self, origin, R):
		rsqrt2 = 1.0/ math.sqrt(2)
		HyperplaneRepresentation.__init__(self, origin,
			[(1,0,-rsqrt2), # vertices of a tetrahedron
			 (-1,0,-rsqrt2),
			 (0,1,rsqrt2),
			 (0,-1,rsqrt2)],
			R / (rsqrt2 + 1/rsqrt2))

class Octahedron(SymmetricDualPolyhedron):
	def __init__(self, origin, R):
		SymmetricDualPolyhedron.__init__(self, origin,
			[(1,1,1), # vertices of a cube
			 (1,-1,1),
			 (-1,1,1),
			 (-1,-1,1)],
			R / 3.0)

class Dodecahedron(SymmetricDualPolyhedron):
	def __init__(self, origin, R):
		phi = 0.5 * (1 + math.sqrt(5)) # the golden ratio
		SymmetricDualPolyhedron.__init__(self, origin,
			[(0,1,phi), # vertices of an icosahedron
			 (1,phi,0),
			 (phi,0,1),
			 (0,-1,phi),
			 (-1,phi,0),
			 (phi,0,-1)],
			R / (phi + 1.0/phi))

class Icosahedron(SymmetricDualPolyhedron):
	def __init__(self, origin, R):
		phi = 0.5 * (1 + math.sqrt(5)) # the golden ratio
		SymmetricDualPolyhedron.__init__(self, origin,
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
