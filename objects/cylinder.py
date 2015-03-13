from objects.components import LayerComponent, CylinderComponent
from objects import ConvexIntersection

class Cylinder(ConvexIntersection):
	
	def __init__(self, bottom_center, axis, height, R):
		components = [ LayerComponent(axis, height), CylinderComponent(axis,R) ]
		ConvexIntersection.__init__(self, bottom_center, components)
		self.unique_tracer_id = ''

