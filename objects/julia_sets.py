
# Tracer objects: Implicit surfaces

from tracer import Tracer
from objects import ImplicitSurface

class QuaternionJuliaSet(ImplicitSurface):
    
    def __init__(self, c, julia_itr, *args, **argd):
        self.c = c
        self.julia_itr = julia_itr
        argd['bndR'] = 1.5
        argd['self_intersection'] = False
        ImplicitSurface.__init__(self, "x^2 + y^2 + z^2 - 1", *args, **argd)
        
    def f_code_template(self):
        return 'objects/QuaternionJuliaSet.cl'
    
    def compute_f_code(self):
        self.c_code = [str(x) for x in self.c]
    
    def compute_df_code(self):
        pass

class QuaternionJuliaSet2(Tracer):
    
    def __init__(self, c, julia_itr, **kwargs):
        self.c = c
        self.julia_itr = julia_itr
        self.bndR = 4.0
        self.center = (0, 0, 0)
        self.max_itr = 100
        self.precision = 0.001
        self.max_step = 0.1
        
        for (k, v) in kwargs.items():
            setattr(self, k, v)
    
    @property
    def c_as_float4(self):
        return "(float4)%s" % (tuple(self.c),)
