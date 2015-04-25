from tracer import Tracer
from utils import normalize
import numpy

class TriangleMesh(Tracer):
    
    def __init__(self, vertices, faces, normals=None, \
                 center=(0,0,0), scale=1.0, \
                 auto_scale=False, auto_flip_normal=False,
                 shading='flat', auto_smooth_normals=False):
        
        center = numpy.reshape(numpy.array(center), (1,3))
        
        self.vertices = to_vector_array(vertices)
        self.faces = to_vector_array(faces, dtype=numpy.int32)
        
        if auto_scale:
            current_center, current_size = self.get_bounding_cube()
            self.vertices = self.vertices - current_center[numpy.newaxis,:]
            
            if current_size > 0.0: scale = scale / float(current_size*0.5)
            
            print "auto-scaling to %.1f%%" % (scale*100)
        
        self.vertices = self.vertices*scale + center
        self.auto_flip_normal = auto_flip_normal
        self.shading = shading
        
        self.unique_tracer_id = '_' + self.shading + '_autoflip_%s' % self.auto_flip_normal
        if self.shading != 'flat':
            if auto_smooth_normals:
                normals = generate_smooth_normals(self.vertices, self.faces)
            
            self.normals = to_vector_array(normals)
            assert(self.normals.shape[0], self.vertices.shape[0])
        else:
            self.normals = None
            assert(normals is None)
        
    def get_data(self):
        vector_data = [self.vertices]
        if self.normals is not None: vector_data.append(self.normals)
        
        return {
            'vector': numpy.vstack(vector_data),
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
        
    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    def parameter_declarations(self):
        return ['int n_vertices', 'int n_triangles']
    
    def parameter_values(self):
        return [self.n_vertices, self.n_faces]

def to_vector_array(array_of_arrays, dtype=numpy.float32):
    vec_array = numpy.array(array_of_arrays, dtype=dtype)
    if len(vec_array.shape) != 2 or vec_array.shape[-1] != 3:
        raise RuntimeError('invalid shape %s' % vec_array.shape)
    return vec_array

def generate_smooth_normals(vertices, faces):
    
    print 'generating normals for', vertices.shape[0], 'vertices'
    
    vertex_normals = [[] for _ in xrange(vertices.shape[0])]
    
    print len(vertex_normals), 'vertices'
    
    normalize = lambda n: n / numpy.sqrt(numpy.sum(n**2))
    
    for i in range(faces.shape[0]):
        face_vertices = faces[i,:]
        v1,v2,v3 = [vertices[face_vertices[j],:] for j in range(3)]
        
        n = normalize(numpy.cross(v2-v1,v3-v1))
        for v in face_vertices:
            vertex_normals[v].append(n)
    
    normals = numpy.ones(vertices.shape)
    for i in range(normals.shape[0]):
        if len(vertex_normals[i]) == 0:
            print "WARNING: no normal for vertex", i
            continue
        avg_normal = numpy.mean(numpy.vstack(vertex_normals[i]),0)
        normals[i,:] = normalize(avg_normal)
    
    return normals
