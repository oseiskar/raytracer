from tracer import Tracer
from utils import normalize
import numpy

class Octree(Tracer):
    
    MAX_DEPTH=7
    
    def __init__(self, triangle_mesh, max_depth=3, max_faces_per_leaf=5):
        self.triangle_mesh = triangle_mesh
        self.max_depth = max_depth
        self.max_faces_per_leaf = max_faces_per_leaf
        if self.max_depth >= self.__class__.MAX_DEPTH:
            raise RuntimeError("max_depth >= MAX_DEPTH")
        self.build()
    
    class Node:
        
        child_order = [[x,y,z] for x in [0,1] for y in [0,1] for z in [0,1]]
        
        def __init__(self, origin, size, parent = None, coordinates = [[0],[0],[0]]):
            self.children = []
            self.faces = []
            self.origin = origin
            self.size = size
            self.coordinates = coordinates
            self.parent = parent
        
        def is_empty(self):
            return len(self.faces) == 0
        
        def is_leaf(self):
            return len(self.children) == 0
        
        def get_bounding_sphere(self):
            return self.center, numpy.sqrt(3)*self.size*0.5
            
        @property
        def center(self):
            return numpy.array([self.size*0.5]*3) + self.origin
        
        def get_children(self):
            if len(self.children) > 0: return self.children
            for xyz in Octree.Node.child_order:
                self.children.append(Octree.Node(\
                    origin = self.origin + numpy.array(xyz)*0.5*self.size, \
                    size = self.size*0.5, \
                    parent = self,
                    coordinates = \
                        [self.coordinates[i] + [xyz[i]] for i in range(3)]))
            return self.children
    
    def build(self):
        
        self._compute_coord_ranges()
        
        self.origin = numpy.array(self.center) - numpy.array([self.size*0.5]*3) 
        self.root = Octree.Node(self.origin, self.size)
        
        face_centers, face_radii = face_bounding_spheres(\
            self.triangle_mesh.vertices, self.triangle_mesh.faces)
        
        self.leaves = []
        
        depth = 0
        active_nodes = [self.root]
        
        while len(active_nodes) > 0:
            
            node_centers = []
            node_radii = []
            for node in active_nodes:
                c, r = node.get_bounding_sphere()
                node_centers.append(c)
                node_radii.append(r)
            node_centers = numpy.array(node_centers)
            node_radii = numpy.array(node_radii)
            
            for node, face in all_intersections(node_centers, node_radii, \
                                                face_centers, face_radii):
                active_nodes[node].faces.append(face)
            
            new_active = []
            for node in active_nodes:
                if len(node.faces) <= self.max_faces_per_leaf \
                        or depth == self.max_depth:
                    self.leaves.append(node)
                else:
                    node.faces = []
                    new_active += node.get_children()
            
            print 'depth', depth, 'active_nodes', len(active_nodes), 'total leaves', len(self.leaves)
            
            active_nodes = new_active
            depth += 1
    
    def _compute_coord_ranges(self):
        
        vertices = self.triangle_mesh.vertices
        faces = self.triangle_mesh.faces
        self.coord_ranges = []
        
        for coord in range(3):
            values = vertices[:,coord]
            self.coord_ranges.append([
                numpy.min(values), numpy.max(values)
            ])
        
        self.center = [sum(x)*0.5 for x in self.coord_ranges]
        self.size = max([abs(x[1]-x[0]) for x in self.coord_ranges])*1.01
        
    def get_data(self):
        """serializes the octree data structure"""
        
        def write_node(node, tree_data):
            child_mask = 0
            if node.is_leaf():
                if node.is_empty():
                    child_mask = 0x100
                    data = []
                else:
                    data = [len(node.faces)] + node.faces
            else:
                data = []
                for i in range(len(node.children)):
                    c = node.children[i]
                    if c.is_empty() and c.is_leaf():
                        node_header = [0x100,0]
                    else:
                        child_mask = child_mask | (0x1 << i)
                        node_header = write_node(c, tree_data)
                    data += node_header
            data_offset = len(tree_data)
            tree_data.extend(data)
            return [child_mask, data_offset]
        
        tree_data = []
        root_data = write_node(self.root, tree_data)
        self.root_data_offset = len(tree_data)
        tree_data.extend(root_data)
        print root_data

        return {
            'vector': self.triangle_mesh.vertices,
            'integer': numpy.concatenate([numpy.ravel(self.triangle_mesh.faces), tree_data])
        }
    
    @property
    def total_faces(self):
        return len(self.triangle_mesh.faces)

def face_bounding_spheres(vertices, faces):
    triangles = numpy.dstack([vertices[faces[:,i], :] for i in range(3)])
    
    centers = numpy.mean(triangles,-1)
    sq_distances = numpy.sum((triangles - centers[...,numpy.newaxis])**2,1)
    radii = numpy.sqrt(numpy.max(sq_distances,-1))
    
    return centers, radii

def all_intersections(node_centers, node_radii, face_centers, face_radii):
    nc = numpy.transpose(node_centers[...,numpy.newaxis], (0,2,1))
    fc = numpy.transpose(face_centers[...,numpy.newaxis], (2,0,1))
    
    dist_mat = numpy.sqrt(numpy.sum((nc-fc)**2,2))
    dist_mat = dist_mat - node_radii[:,numpy.newaxis] - face_radii[numpy.newaxis,:]
    
    return numpy.transpose(numpy.nonzero(dist_mat < 0))
    
