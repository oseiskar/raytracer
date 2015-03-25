
from objects import *
import utils
import numpy as np
import shader
from spectrum import Spectrum

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

class Scene:
    
    def get_camera_rotmat(self):
        return utils.camera_rotmat(self.camera_direction, self.camera_up)
    
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
        return utils.camera_rays(self.image_size, self.camera_flat_ccd, \
            self.camera_fov, self.camera_direction, self.camera_up)
    
    def get_number_of_camera_rays(self):
        return self.get_camera_rays().size / 3
    
    def add_object(self, tracer, material, name=None):
        self.objects.append(Object(tracer, material, name))

    def get_kernels(self, template_env):
        kernel_map = {}
        for obj in self.objects:
            for (k, v) in obj.tracer.make_functions(template_env).items():
                if k in kernel_map and kernel_map[k] != v:
                    print kernel_map[k]
                    print '------'
                    print v
                    raise RuntimeError("kernel name clash!!")
                kernel_map[k] = v
        return list(set(kernel_map.values()))
    
    def collect_vector_data(self):
        datas = []
        n_data = 0
        for obj in self.objects:
            if obj.tracer.has_vector_data():
                cur_data = np.array(obj.tracer.get_vector_data())
                if cur_data.shape[1] != 3:
                    raise RuntimeError('invalid vector data shape')
                datas.append(np.array(cur_data))
                obj.vector_data_offset = n_data
                cur_data_len = cur_data.shape[0]
            else:
                cur_data_len = 0
            
            obj.vector_data_len = cur_data_len
            n_data += cur_data_len
            
        if len(datas) == 0: return None
        return np.vstack(datas)

from scenes.default_scenes import DefaultBoxScene, DefaultSpectrumBoxScene
