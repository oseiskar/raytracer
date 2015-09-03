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
