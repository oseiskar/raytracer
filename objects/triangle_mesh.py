from tracer import Tracer
from utils import normalize
import numpy

class TriangleMesh(Tracer):
    
    def __init__(self, vertices, faces, center=(0,0,0), scale=1.0, \
                 auto_scale=False, auto_normal=False):
        
        center = numpy.reshape(numpy.array(center), (1,3))
        
        vertices = numpy.array(vertices)
        self.faces = numpy.array(faces, dtype=numpy.int32)
        
        if self.faces.shape[1] != 3 or vertices.shape[1] != 3:
            raise RuntimeError('invalid shape')
        
        if auto_scale:
            vertices = vertices - numpy.mean(vertices)
            maxabs = numpy.max(numpy.ravel(numpy.abs(vertices)))
            if maxabs > 0.0: scale = scale / float(maxabs)
        
        self.vertices = vertices*scale + center
        
        self.auto_flip_normal = auto_normal
        
    def get_data(self):
        return {
            'vector': self.vertices,
            'integer': numpy.ravel(self.faces)
        }
    
    @property
    def n_faces(self):
        return self.faces.shape[0]
