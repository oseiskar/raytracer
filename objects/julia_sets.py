
# Tracer objects: Implicit surfaces

from tracer import Tracer
from utils import normalize_tuple, vec_norm
import numpy
from objects import Sphere, ImplicitSurface

class QuaternionJuliaSet(ImplicitSurface):
    
    def __init__(self, c, julia_itr, *args, **argd):
        self.c = c
        self.julia_itr = julia_itr
        argd['bndR'] = 1.5
        argd['self_intersection'] = False
        ImplicitSurface.__init__(self, "x^2 + y^2 + z^2 - 1", *args, **argd)
    
    def compute_f_code(self):
        
        return """
        ia_type qr = x - %s;
        ia_type qi = y - %s;
        ia_type qj = z - %s;
        qr /= %s;
        qi /= %s;
        qj /= %s;
        ia_type qk = ia_new(0,0);
        
        const float cr = %s, ci = %s, cj = %s, ck = %s;
        ia_type qr1;
        
        for (int iii=0; iii<%d; ++iii)
        {
            // Quaternion operation z -> z^2 + c
            // lazy... "should" use ia_add
            qr1 = ia_sub(ia_pow2(qr), ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk))+cr;
            qi = 2 * ia_mul(qr,qi) + ci;
            qj = 2 * ia_mul(qr,qj) + cj;
            qk = 2 * ia_mul(qr,qk) + ck;
            qr = qr1;
        }
        
        f = ia_pow2(qr)+ia_pow2(qi)+ia_pow2(qj)+ia_pow2(qk) - 4.0;
        """ % (self.center+tuple([self.scale]*3)+self.c+(self.julia_itr,))
        
        return "f = %s;" % self.eq
    
    def compute_df_code(self): # TODO
        return """
            df = 1;
        """
    
    def compute_normal_code(self):
        return """
        
        float qr = pos.x - %s;
        float qi = pos.y - %s;
        float qj = pos.z - %s;
        float qk = 0;
        qr /= %s;
        qi /= %s;
        qj /= %s;
        
        float3
            gr = (float3)(1,0,0),
            gi = (float3)(0,1,0),
            gj = (float3)(0,0,1),
            gk = (float3)(0,0,0),
            gr1, gi1, gj1, gk1;
        
        const float cr = %s, ci = %s, cj = %s, ck = %s;
        float qr1;
        
        for (int iii=0; iii<%d; ++iii)
        {
            // Derivative chain rule...
            
            gr1 = 2*(qr*gr - qi*gi - qj*gj - qk*gk);
            gi1 = 2*(gr*qi + qr*gi);
            gj1 = 2*(gr*qj + qr*gj);
            gk1 = 2*(gr*qk + qr*gk);
            
            // Quaternion operation z -> z^2 + c
            qr1 = qr*qr - qi*qi - qj*qj - qk*qk + cr;
            qi = 2 * qr*qi + ci;
            qj = 2 * qr*qj + cj;
            qk = 2 * qr*qk + ck;
            qr = qr1;
            
            gr = gr1;
            gi = gi1;
            gj = gj1;
            gk = gk1;
        }
        
        //const float3 grad = gr*qr;
        float3 grad = fast_normalize(2*(qr*gr + qi*gi + qj*gj + qk*gk));
        
        if (all(isfinite(grad))) *p_normal = grad;
        else *p_normal = (float3)(1,0,0);
        
        //const float l = length(grad);
        //if (l==0) *p_normal = (1,0,0);
        //else *p_normal = grad/l;
        //*p_normal = (float3)(1,0,0);
        //*p_normal = fast_normalize((float3)(1,gr.x*gr.x*0.001,0));
        
        //*p_normal = fast_normalize((float3)(dx, dy, dz));
        """ % (self.center+tuple([self.scale]*3)+self.c+(self.julia_itr,))



class QuaternionJuliaSet2(Tracer):
    
    def __init__(self, c, julia_itr, **kwargs):
        self.c = c
        self.julia_itr = julia_itr
        self.bndR = 4.0
        self.center = (0,0,0)
        self.max_itr = 100
        self.precision = 0.001
        self.max_step = 0.1
        
        # TODO: tuple(center)
        
        for (k,v) in kwargs.items(): setattr(self, k, v)
    
    @property
    def c_as_float4(self):
        return "(float4)%s" % (tuple(self.c),)
