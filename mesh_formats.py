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
            if face_vertices[0] != len(face_vertices)-1:
                raise "invalid format"
            
            faces += list(poly_to_triangle_fan(face_vertices[1:]))
            
        print "constructed", len(faces), "triangles"
        return vertices, faces


