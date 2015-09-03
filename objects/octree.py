from tracer import Tracer
import numpy

class Octree(Tracer):
    
    MAX_DEPTH=7
    
    def __init__(self, triangle_mesh, max_depth=3, max_faces_per_leaf=5):
        
        original_coordinates = triangle_mesh.coordinates
        self.triangle_mesh = triangle_mesh
        Tracer.__init__(self)
        self.coordinates = original_coordinates
        
        self.max_depth = max_depth
        self.max_faces_per_leaf = max_faces_per_leaf
        if self.max_depth >= self.__class__.MAX_DEPTH:
            raise RuntimeError("max_depth >= MAX_DEPTH")
        self.build()
    
    @property
    def auto_flip_normal(self):
        return self.triangle_mesh.auto_flip_normal
    
    @property
    def coordinates(self):
        return self.triangle_mesh.coordinates
    
    @coordinates.setter
    def coordinates(self, var):
        self.triangle_mesh.coordinates = var
    
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
        
        self.center, self.size = self.triangle_mesh.get_bounding_cube()
        self.size *= 1.01
        
        self.origin = numpy.array(self.center) - numpy.array([self.size*0.5]*3) 
        self.root = Octree.Node(self.origin, self.size)
        
        face_centers, face_radii = face_bounding_spheres(\
            self.triangle_mesh.vertices, self.triangle_mesh.faces)
        
        face_normals, face_h = face_planes(\
            self.triangle_mesh.vertices, self.triangle_mesh.faces)
        
        self.leaves = []
        
        depth = 0
        active_nodes = [self.root]
        active_faces = range(self.triangle_mesh.n_faces)
        
        while len(active_nodes) > 0:
            
            intersections, active_faces = all_intersections(\
                active_nodes, active_faces, \
                self.triangle_mesh.faces, self.triangle_mesh.vertices,
                face_centers, face_radii, face_normals, face_h)
            
            for node, face_idx in intersections:
                node.faces.append(face_idx)
            
            new_active = []
            for node in active_nodes:
                n_smaller_faces = len([f for f in node.faces \
                    if face_radii[f] < node.size*0.5])
                
                if n_smaller_faces <= self.max_faces_per_leaf \
                        or depth == self.max_depth:
                    self.leaves.append(node)
                else:
                    node.faces = []
                    new_active += node.get_children()
            
            print 'depth:', depth, \
                'active nodes:', len(active_nodes), \
                'total leaves:', len(self.leaves), \
                'active faces:', len(active_faces)
            
            active_nodes = new_active
            depth += 1
        
    def get_data(self):
        """serializes the octree data structure"""
        
        def write_node(node, tree_data):
            child_mask = 0
            if node.is_leaf():
                if node.is_empty():
                    return [0x100, -99999]
                else:
                    data = [len(node.faces)] + node.faces
            else:
                data = []
                for i in range(len(node.children)):
                    c = node.children[i]
                    if c.is_empty() and c.is_leaf():
                        node_header = [0x100, -99999]
                    else:
                        child_mask = child_mask | (0x1 << i)
                        node_header = write_node(c, tree_data)
                        #assert(node_header[0] != 0x100)
                    data += node_header
                if child_mask == 0: return [0x100, -99999]
            data_offset = len(tree_data)
            tree_data.extend(data)
            return [child_mask, data_offset]
        
        tree_data = []
        root_data = write_node(self.root, tree_data)
        self.root_data_offset = len(tree_data)
        tree_data.extend(root_data)
        
        mesh_data = self.triangle_mesh.get_data()
        mesh_data['integer'] = numpy.concatenate([
            numpy.ravel(mesh_data['integer']),
            tree_data
        ])
        
        return mesh_data
    
    @property
    def total_faces(self):
        return len(self.triangle_mesh.faces)
        
    def parameter_declarations(self):
        return [
            'int n_vertices',
            'int n_triangles',
            'int root_data_offset',
            'int face_data_length',
            'float3 root_origin',
            'float root_size']
    
    def parameter_values(self):
        return [
            self.triangle_mesh.n_vertices,
            self.triangle_mesh.n_faces,
            self.root_data_offset,
            self.total_faces*3,
            self.root.origin,
            self.root.size]

def face_bounding_spheres(vertices, faces):
    triangles = numpy.dstack([vertices[faces[:,i], :] for i in range(3)])
    
    centers = numpy.mean(triangles,-1)
    sq_distances = numpy.sum((triangles - centers[...,numpy.newaxis])**2,1)
    radii = numpy.sqrt(numpy.max(sq_distances,-1))
    
    return centers, radii

def face_planes(vertices, faces):
    tri_vertices = ([vertices[faces[:,i], :] for i in range(3)])
    
    p0 = tri_vertices[0]
    e1 = tri_vertices[1] - p0
    e2 = tri_vertices[2] - p0
    
    normals = numpy.cross(e1,e2)
    normals = normals / numpy.sqrt(numpy.sum(normals**2,1))[:,numpy.newaxis]
    plane_h = numpy.sum(p0*normals,1)
    return (normals, plane_h)

def all_intersections(active_nodes, active_faces, \
    faces, vertices, \
    face_centers, face_radii, \
    face_normals, face_h):
    
    node_centers = []
    node_radii = []
    for node in active_nodes:
        c, r = node.get_bounding_sphere()
        node_centers.append(c)
        node_radii.append(r)
    node_centers = numpy.array(node_centers)
    node_radii = numpy.array(node_radii)
    
    nc = numpy.transpose(node_centers[...,numpy.newaxis], (0,2,1))
    fc = numpy.transpose(face_centers[...,numpy.newaxis], (2,0,1))
    fn = numpy.transpose(face_normals[...,numpy.newaxis], (2,0,1))
    
    dist_mat = numpy.sqrt(numpy.sum((nc-fc)**2,2))
    dist_mat = dist_mat - node_radii[:,numpy.newaxis] - face_radii[numpy.newaxis,:]
    
    dot_prod_mat = numpy.sum(nc*fn,2)
    dist_mat = numpy.maximum(dist_mat, \
        numpy.abs(dot_prod_mat - face_h[numpy.newaxis,:]) - node_radii[:,numpy.newaxis])
    
    isecs = numpy.transpose(numpy.nonzero(dist_mat < 0))
    
    isecs = [(active_nodes[ni], active_faces[fi])  for ni, fi in isecs]
    
    any_intersections = numpy.any(dist_mat < 0, 0)
    active_faces = numpy.array(active_faces)[any_intersections]
    
    return (isecs, active_faces)
