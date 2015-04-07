# 10-minute implementations of reading different mesh formats

def poly_to_triangle_fan(vertices):
    for offset in range(len(vertices)-2):
        yield([vertices[0],vertices[offset+1],vertices[offset+2]])

def read_off(filename):
    with open(filename, 'r') as off:
        
        first_line = off.readline().strip()
        none, sep, rest = first_line.partition('OFF')
        if len(none) > 0 or sep != 'OFF':
            raise RuntimeError('not OFF format')
        rest = rest.strip()
        if len(rest) == 0:
            rest = off.readline().strip()
        n_points, n_faces, _ = [int(x) for x in rest.split()]
        
        print 'reading OFF with', n_points, 'points and', n_faces, 'faces'
        
        vertices = []
        for v in xrange(n_points):
            vertices.append([float(x) for x in off.readline().split()])
        
        faces = []
        for f in xrange(n_faces):
            face_vertices = [int(x) for x in off.readline().split()]
            assert(face_vertices[0] == len(face_vertices)-1)
            
            faces += list(poly_to_triangle_fan(face_vertices[1:]))
            
        print "constructed", len(faces), "triangles"
        return vertices, faces

def read_zipper(filename):
    """the Stanford bunny is published in this format"""
    
    with open(filename, 'r') as zipper:
        
        while True:
            line = zipper.readline().strip().split()
            if len(line) == 0: raise RuntimeError("unexpected eof/empty line")
            if line[0] == 'element':
                if line[1] == 'vertex':
                    n_vertices = int(line[2])
                elif line[1] == 'face':
                    n_faces = int(line[2])
            elif line[0] == 'end_header':
                break
        
        vertices = []
        faces = []
        
        for i in xrange(n_vertices):
            line = zipper.readline().strip().split()
            vertices.append([float(line[c]) for c in range(3)])
        
        for i in xrange(n_faces):
            face_vertices = [int(x) for x in zipper.readline().strip().split()]
            assert(face_vertices[0] == len(face_vertices)-1)
            faces += list(poly_to_triangle_fan(face_vertices[1:]))
    
        return vertices, faces

def remove_duplicate_faces(faces, verbose=False):
    face_set = set([])
    new_faces = []
    for face in faces:
        f = frozenset(face)
        if f in face_set:
            if verbose: print "WARNING: removed duplicate face %s" % face
        else:
            face_set.add(f)
            new_faces.append(face)
    return new_faces
