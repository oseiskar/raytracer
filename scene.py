
from objects import *
import utils
import numpy as np

class Object:
	"""
	An object consists of a Tracer that represents its shape and a material
	"""
	def __init__(self, tracer, material, name=None):
		self.tracer = tracer
		self.material = material
		self.name = name

# Default scene
class Scene:
	"""
	Default raytracer scene. Override attributes as required
	"""
	
	# helpers...
	
	@staticmethod
	def make_world_box( material, dims, center=(0,0,0) ):
		return [\
			Object(HalfSpace( ( 1, 0, 0), dims[0]-center[0] ), material, 'wall'), \
			Object(HalfSpace( (-1, 0, 0), dims[0]+center[0] ), material, 'wall'), \
			Object(HalfSpace( ( 0, 1, 0), dims[1]-center[1] ), material, 'wall'), \
			Object(HalfSpace( ( 0,-1, 0), dims[1]+center[1] ), material, 'wall'), \
			Object(HalfSpace( ( 0, 0, 1), dims[2]-center[2] ), material, 'floor'), \
			Object(HalfSpace( ( 0, 0,-1), dims[2]+center[2] ), material, 'ceiling')]
	
	def get_camera_rotmat(self):
		return utils.camera_rotmat(self.camera_direction, self.camera_up)
	
	def get_objects(self,name):
		return [obj for obj in self.objects if obj.name == name]
	
	def get_object(self,name):
		objs = self.get_objects(name)
		if len(objs) == 1: return objs[0]
		elif len(objs) == 0:
			raise KeyError("No object named '%s'" % name)
		else:
			raise KeyError("Multiple objects in the scene are called '%s'" % name)
	
	def delete_objects(self,name):
		self.objects[:] = [obj for obj in self.objects if obj.name != name]
	
	def direct_camera_towards(self, target):
		self.camera_direction = np.array(target)-np.array(self.camera_position)
	
	def get_camera_rays(self):
		return utils.camera_rays(self.image_size, self.camera_flat_ccd, \
			self.camera_fov, self.camera_direction, self.camera_up)
    
	def get_number_of_camera_rays(self):
		return self.get_camera_rays().size / 3
	
	def __init__(self):
		"""Initialize default scene"""
		
		# --- Image settings
		self.image_size = (800,600)
		self.brightness = 0.3
		self.gamma = 1.8
		
		# --- Raytracer settings
		self.tent_filter = True
		self.quasirandom = True
		self.samples_per_pixel = 10000
		self.min_bounces = 3
		self.russian_roulette_prob = .3
		self.max_bounces = 4
		
		# --- Materials
		self.materials = {\
			'default': # "Air" / initial / default material
				{ 'diffuse': ( 1, 0, 1),
				  'emission':(0, 0, 0),
				  'reflection':(0,0,0),
				  'transparency': (0,0,0),
				  'ior': (1.0,), # Index Of Refraction
                  'dispersion': (0.0,),
				  'vs': (0,0,0) # "fog"
				}, 
				# Other materials
			'white':
				{ 'diffuse': ( .8, .8, .8) }, 
			'green':
				{ 'diffuse': (0.4,0.9,0.4)},
			'red':
				{ 'diffuse': (.7,.4,.4) }, 
			'mirror':
				{ 'diffuse': (.2,.2,.2), 'reflection':(.7,.7,.7) },
			'light': # warm yellow-orange-light
				{ 'diffuse': ( 1, 1, 1), 'emission':(4,2,.7) },
			'sky':
				{ 'diffuse': ( 0, 0, 0), 'emission':(.5,.5,.7) },
			'glass':
				{ 'diffuse': (.1,.1,.1), 'transparency':(.7,.7,.7), 'reflection':(.2,.2,.2), 'ior':(1.5,) },
			}
		
		# --- Objects
		self.objects = Scene.make_world_box( 'white', (3,5,2), (0,0,2) )
		self.objects[-1].material = "sky" # world box ceiling
		self.objects[-2].material = "green" # world box floor
		self.root_object = None
		self.max_ray_length = 100
        
		self.shader = 'rgb_shader'
		
		# light bulb on the right wall
		self.objects.append(Object(Sphere( (-3,-1,2), 0.5 ), 'light', 'light'))
		
		# --- Camera
		self.camera_up = (0,0,1)
		self.camera_position = (1,-5,2)
		self.camera_fov = 55 # Field-of-view angle (horizontal)
		self.camera_flat_ccd = False
		camera_target = (0,2,0.5)
		self.camera_dof_fstop = 0.0
		self.camera_sharp_distance = 0.0
		self.direct_camera_towards(camera_target)
		

scene = Scene()
