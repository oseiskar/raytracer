from objects.components import LayerComponent, CylinderComponent
from objects import ConvexIntersection

class Cone(ConvexIntersection):
	
	def __init__(self, tip, axis, height, R):
		components = [ HalfSpaceComponent(axis, height), ConeComponent( (0,0,0), axis, R / float(height)) ]
		ConvexIntersection.__init__(self, tip, components)
		self.unique_tracer_id = ''
