import numpy

EPSILON = 1e-8

class Affine:
    def __init__(self,
            linear=numpy.identity(3),
            translation=numpy.zeros(3),
            rotation_axis=None,
            rotation_deg=None,
            scaling=None):
        
        self.translation = numpy.ravel(translation)
        if rotation_axis is not None:
            rot = rotation_matrix(rotation_axis, rotation_deg)
            linear = numpy.dot(rot, linear)
        
        if scaling is not None:
            scaling = numpy.ravel(scaling)
            if len(scaling) == 3:
                scaling = numpy.reshape(scaling, (3,1))
            linear = scaling * linear
        
        self.linear = linear
    
    def __call__(self, other):
        if isinstance(other, Affine):
            return Affine( \
                numpy.dot(self.linear, other.linear), \
                numpy.dot(self.linear, other.translation) + self.translation )
        else:
            return numpy.dot(self.linear, numpy.ravel(other)) + self.translation

    def inverse(self):
        linear_inv = numpy.linalg.inv(self.linear)
        inv_trans = -numpy.dot(linear_inv, self.translation)
        return Affine( linear_inv, inv_trans )
    
    def is_identity(self, epsilon=EPSILON):
        mat_norm = numpy.linalg.norm(self.linear - numpy.identity(3), 'fro')
        vec_norm = numpy.linalg.norm(self.translation)
        return mat_norm + vec_norm < epsilon
    
    def transpose(self):
        return Affine(self.linear.T, -self.translation)
    
    def is_orthogonal(self, epsilon=EPSILON):
        return self.transpose()(self).is_identity()
    
    @staticmethod
    def identity():
        return Affine()
    
    def __str__(self):
        return "AffineTransformation(\n%s * x +\n%s)" \
            % (self.linear, self.translation)

def axis_symbol_to_vector(letter):
    letter = letter.lower()
    if letter == 'x': return (1,0,0)
    if letter == 'y': return (0,1,0)
    if letter == 'z': return (0,0,1)
    else:
        raise RuntimeError("invalid axis letter %s" % letter)
    
def rotation_matrix(axis, angle_deg):
    
    angle_rad = numpy.pi / 180.0 * angle_deg
    if isinstance(axis, str):
        axis = axis_symbol_to_vector(axis)
    
    axis = numpy.ravel(axis)
    axis = axis * (1.0 / numpy.linalg.norm(axis))
    
    def rodrigues(vec):
        # Rodrigues rotation formula
        vec = numpy.ravel(vec)
        c = numpy.cos(angle_rad)
        s = numpy.sin(angle_rad)
        cross = numpy.cross(axis, vec)
        dot = numpy.dot(axis, vec)
        return c * vec + s * cross + axis * dot * (1.0 - c)
    
    return numpy.vstack([
        rodrigues((1,0,0)),
        rodrigues((0,1,0)),
        rodrigues((0,0,1))
    ]).T



