from tracer import Tracer
from utils import normalize
import numpy

class TriangleMesh(Tracer):
    
    def __init__(self, vertices, faces, center=(0,0,0), scale=1.0, \
                 auto_scale=False, auto_normal=False):
        
        center = numpy.reshape(numpy.array(center), (1,3))
        
        self.vertices = numpy.array(vertices)
        self.faces = numpy.array(faces, dtype=numpy.int32)
        
        if self.faces.shape[1] != 3 or self.vertices.shape[1] != 3:
            raise RuntimeError('invalid shape')
            
        if auto_scale:
            current_center, current_size = self.get_bounding_cube()
            self.vertices = self.vertices - current_center[numpy.newaxis,:]
            
            if current_size > 0.0: scale = scale / float(current_size*0.5)
            
            print "auto-scaling to %.1f%%" % (scale*100)
        
        self.vertices = self.vertices*scale + center
        
        self.auto_flip_normal = auto_normal
        
    def get_data(self):
        return {
            'vector': self.vertices,
            'integer': numpy.ravel(self.faces)
        }
        
    def get_bounding_box(self):
        coord_ranges = []
        for coord in range(3):
            values = self.vertices[:,coord]
            coord_ranges.append([
                numpy.min(values), numpy.max(values)
            ])
        return coord_ranges
    
    def get_bounding_cube(self):
        
        coord_ranges = self.get_bounding_box()
        
        center = [sum(x)*0.5 for x in coord_ranges]
        side = max([abs(x[1]-x[0]) for x in coord_ranges])
        return numpy.array(center), side
    
    @property
    def n_faces(self):
        return self.faces.shape[0]
