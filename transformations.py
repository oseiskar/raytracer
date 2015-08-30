import numpy as np

EPSILON = 1e-8

class Affine:
    def __init__(self,
            linear=None,
            translation=None,
            rotation_axis=None,
            rotation_deg=None,
            scaling=None):
        
        linear = linear if linear is not None else np.identity(3)
        translation = translation if translation is not None else np.zeros(3)
        
        self._translation = np.ravel(translation)
        if rotation_axis is not None:
            rot = rotation_matrix(rotation_axis, rotation_deg)
            linear = np.dot(rot, linear)
        
        if scaling is not None:
            scaling = np.ravel(scaling)
            if len(scaling) == 3:
                scaling = np.reshape(scaling, (3,1))
            linear = scaling * linear
        
        self._linear = linear
    
    def __call__(self, other):
        if isinstance(other, Affine):
            return Affine( \
                np.dot(self._linear, other._linear), \
                np.dot(self._linear, other._translation) + self._translation )
        else:
            return np.dot(self._linear, np.ravel(other)) + self._translation

    def inverse(self):
        linear_inv = np.linalg.inv(self._linear)
        inv_trans = -np.dot(linear_inv, self._translation)
        return Affine( linear_inv, inv_trans )
    
    def is_identity(self, epsilon=EPSILON):
        mat_norm = np.linalg.norm(self._linear - np.identity(3), 'fro')
        vec_norm = np.linalg.norm(self._translation)
        return mat_norm + vec_norm < epsilon
    
    def transpose(self):
        return Affine(self._linear.T, -self._translation)
    
    def is_orthogonal(self, epsilon=EPSILON):
        return self.transpose()(self).is_identity()
    
    @property
    def linear(self):
        return self._linear*1.0
        
    @property
    def translation(self):
        return self._translation*1.0
    
    @staticmethod
    def identity():
        return Affine()
    
    def __str__(self):
        return "AffineTransformation(\n%s * x +\n%s)" \
            % (self._linear, self._translation)

def axis_symbol_to_vector(letter):
    letter = letter.lower()
    if letter == 'x': return (1,0,0)
    if letter == 'y': return (0,1,0)
    if letter == 'z': return (0,0,1)
    else:
        raise RuntimeError("invalid axis letter %s" % letter)
    
def rotation_matrix(axis, angle_deg):
    
    angle_rad = np.pi / 180.0 * angle_deg
    if isinstance(axis, str):
        axis = axis_symbol_to_vector(axis)
    
    axis = np.ravel(axis)
    axis = axis * (1.0 / np.linalg.norm(axis))
    
    def rodrigues(vec):
        # Rodrigues rotation formula
        vec = np.ravel(vec)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        cross = np.cross(axis, vec)
        dot = np.dot(axis, vec)
        return c * vec + s * cross + axis * dot * (1.0 - c)
    
    return np.vstack([
        rodrigues((1,0,0)),
        rodrigues((0,1,0)),
        rodrigues((0,0,1))
    ]).T



