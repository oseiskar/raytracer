from tracer import Tracer

class DistanceField(Tracer):
    
    def __init__(self, tracer_code, normal_code=None,
        center=(0,0,0), bndR=None, max_itr=1000,
        precision=1e-5, self_intersection=True):
        
        self.center = center
        self.max_itr = max_itr
        self.precision = precision
        self.no_self_intersection = not self_intersection
        self.tracer_code = tracer_code.strip()
        self.normal_code = normal_code
        
        self.unique_tracer_id = str(id(self))
