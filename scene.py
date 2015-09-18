import numpy as np
import camera
from imgutils import EncodingSettings

class Scene:
    """
    Defines a 3D scene consisting of a camera, objects, materials and
    some rendering settings such as image size
    """
    
    class Object:
        """
        An object consists of a Tracer that represents its shape and a material.
        It is also defined here if the object is to be used as a light source
        in bidirectional path tracing. If yes, this must be supported by the
        Tracer.
        """
        def __init__(self, tracer, material, name=None):
            self.tracer = tracer
            self.material = material
            self.name = name
            self.bidirectional_light = False
    
    class ImageSettings(EncodingSettings):
        pass
    
    def get_camera_rotmat(self):
        return camera.camera_rotmat(self.camera_direction, self.camera_up)
    
    def get_objects(self, name):
        return [obj for obj in self.objects if obj.name == name]
    
    def get_object(self, name):
        objs = self.get_objects(name)
        if len(objs) == 1: return objs[0]
        elif len(objs) == 0:
            raise KeyError("No object named '%s'" % name)
        else:
            raise KeyError("Multiple objects in the scene are called '%s'" % name)
    
    def delete_objects(self, name):
        self.objects[:] = [obj for obj in self.objects if obj.name != name]
    
    def direct_camera_towards(self, target):
        self.camera_direction = np.array(target)-np.array(self.camera_position)
        self.camera_sharp_distance = np.linalg.norm(self.camera_direction)
    
    def get_camera_rays(self):
        return camera.camera_rays(self.image.size, self.camera_flat_ccd, \
            self.camera_fov, self.camera_direction, self.camera_up)
    
    def get_number_of_camera_rays(self):
        return self.get_camera_rays().size / 3
    
    def add_object(self, tracer, material, name=None):
        obj = Scene.Object(tracer, material, name)
        self.objects.append(obj)
        return obj
