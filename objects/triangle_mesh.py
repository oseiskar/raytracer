from tracer import Tracer
from utils import normalize
import numpy

class TriangleMesh(Tracer):
    
    def __init__(self, triangle_data, center=(0,0,0), scale=1.0, auto_scale=False, auto_normal=False):
        center = numpy.reshape(numpy.array(center), (1,3))
        
        arr = numpy.array(triangle_data)
        if auto_scale:
            arr = arr - numpy.mean(arr)
            maxabs = numpy.max(numpy.ravel(numpy.abs(arr)))
            if maxabs > 0.0: scale = scale / float(maxabs)
        
        self.triangle_data = arr*scale + center
        self.auto_flip_normal = auto_normal
        
    def get_vector_data(self):
        return self.triangle_data
    
    @property
    def n_faces(self):
        return self.triangle_data.shape[0]
