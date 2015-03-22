
import numpy as np

def normalize(vecs):
    lens = np.sum(vecs**2, len(vecs.shape)-1)
    lens = np.sqrt(lens)
    lens = np.array(lens)
    lens.shape += (1, )
    lens[lens > 0] = 1.0 / lens[lens > 0]
    return vecs * lens

def normalize_tuple(vec):
    return tuple(normalize(np.array(tuple(vec))))

def vec_norm(vec):
    return np.sqrt(sum([x**2 for x in vec]))

def camera_rotmat(direction, up=(0, 0, 1)):
    
    # TODO camera roll
    
    direction = np.array(direction)
    up = np.array(up)
    
    right = np.cross(direction, up)
    up = np.cross(right, direction)
    
    rotmat = np.vstack((right, -up, direction))
    
    return normalize(rotmat).transpose()

def rotmat_tilt_camera(xa, ya):
    
    rm2d = lambda a : np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    
    rot3dy = np.identity(3)
    rot3dy[1:, 1:] = rm2d(ya)
    
    rot3dx = np.identity(3)
    
    rot3dx[[[0], [2]], [0, 2]] = rm2d(xa)
    
    return np.dot(rot3dy, rot3dx)

def camera_rays(wh, flat=False, wfov=60, direction=(0, 1, 0), up=(0, 0, 1)):
    
    w, h = wh
    wfov = wfov/180.0*np.pi
    aspect = h / float(w)
    hfov = wfov * aspect
    
    rotmat = camera_rotmat(direction, up)
    
    #tilt = rotmat_tilt_camera(0.3,0.4)
    #rotmat = np.dot(rotmat, tilt)
    
    if flat:
        ra = np.tan(wfov * 0.5)
        xr = np.linspace(-ra, ra, w)
        yr = np.linspace(-ra*aspect, ra*aspect, h)
        X, Y = np.meshgrid(xr, yr)
        Z = np.ones(X.shape)
    else:
        pixel_angle = float(wfov)/w
        xa = (np.arange(0, w)+0.5)*pixel_angle - wfov/2.0
        ya = (np.arange(0, h)+0.5)*pixel_angle - hfov/2.0
        Xa, Ya = np.meshgrid(xa, ya)
        
        X = np.sin(Xa)*np.cos(Ya)
        Z = np.cos(Xa)*np.cos(Ya)
        Y = np.sin(Ya)
    
    N = w*h
    vecs = np.dstack((X, Y, Z))
    vecs = np.reshape(vecs, (N, 3)).transpose()
    vecs = np.dot(rotmat, vecs).transpose()
    vecs = np.reshape(vecs, (h, w, 3))
    
    if flat:
        vecs = normalize(vecs)
    
    return vecs

def quasi_random_direction_sample(n, hemisphere=True):
    """
    "Vogel's method / Fermat's spiral",
    adapted from http://blog.marmakoide.org/?p=1
    
    It appears that, in addition to the discussion in the above blog,
    the first n of 2n of these points are evenly distributed in the real
    projective space RP^2 which is very nice.
    
    Now ANY diffusion hemisphere is sampled with a regular even grid!
    """
    
    if hemisphere:
        n = n*2
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)
    
    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z
    
    if hemisphere:
        n = n/2
        points = points[:n, :]
    
    return points

def random_dof_sample():
    
    insideDof = lambda x, y: (x**2 + y**2) < 1.0

    while True:
        dofx = (np.random.rand()-0.5)*2.0
        dofy = (np.random.rand()-0.5)*2.0
        if insideDof( dofx, dofy ):
            return (dofx, dofy)
