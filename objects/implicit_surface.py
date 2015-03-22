
# Tracer objects: Implicit surfaces

from tracer import *
from utils import normalize_tuple, vec_norm
import numpy
from objects import Sphere

class ImplicitSurface(Tracer):

    # The equation defining the surface must be positive outside the object
    # (multiply eq. by -1 if things do not work)
    
    def __init__(self, eq,
                center=(0,0,0), scale=1.0, bndR=None,
                max_itr=1500, precision=0.001, self_intersection=True):
        
        self.unique_tracer_id = str(id(self))
        
        self.center = tuple(center)
        self.scale = scale
        self.no_self_intersection = not self_intersection
        self.precision = precision
        self.max_itr = max_itr
        self.bndR = bndR
        
        import sympy
        import sympy.core.numbers
        
        xyz = sympy.symbols('x y z')
        t = sympy.symbols('t')
        ray = sympy.symbols('ray_x ray_y ray_z')
        origin = sympy.symbols('origin_x origin_y origin_z')
        pos = sympy.symbols('pos.x pos.y pos.z')
        
        self.eq = sympy.sympify(eq)
        
        self.scaled_and_shifted = self.eq.subs([
            (xyz[i], (xyz[i] - self.center[i])/self.scale) \
            for i in range(3) ])
        
        self.ray_paramd = self.scaled_and_shifted.subs([
            (xyz[i], ray[i]*t + origin[i]) \
            for i in range(3) ])
            
        scaled_and_shifted_pos = \
            self.scaled_and_shifted.subs([(xyz[i], pos[i]) for i in range(3)])
        
        self.gradient = [sympy.diff(scaled_and_shifted_pos, pos[i]) for i in range(3)]
        self.derivative = sympy.diff(self.ray_paramd,t)
        
        # Must replace some expressions to make them OpenCL
        class Printer(sympy.printing.str.StrPrinter):
            def _print_Pow(self, expr):
                return ImplicitSurface.print_pow(expr)
        
        old_ptr = sympy.Basic.__str__
        
        sympy.Basic.__str__ = lambda self: Printer().doprint(self)
        
        # Print as an interval arithmetic macro expression
        class IAPrinter(sympy.printing.str.StrPrinter):
            
            def _print_Pow(self, expr):
                base = expr.args[0]
                exponent = expr.args[1]
                return "ia_pow%d(%s)" % (int(exponent), base)
                
            
            def _print_mul_rec(self,args):
                
                a = args[0]
                b = args[1]
                
                if b.is_number: a,b = b,a
                
                if len(args) > 2:
                    bstr = self._print_mul_rec(args[1:])
                else:
                    bstr = str(b)
                
                if a.is_number: 
                    if b.is_number: return "((%s)*(%s))" % (a,b)
                    if a >= 0:
                        return "ia_mul_pos_exact(%s,%s)" % (bstr,a)
                    else:
                        return "ia_mul_neg_exact(%s,%s)" % (bstr,a)
                else:
                    return "ia_mul(%s,%s)" % (a,bstr)
            
            def _print_Mul(self, expr):
                return self._print_mul_rec(expr.args)
                
        sympy.Basic.__str__ = lambda self: IAPrinter().doprint(self)
        self.f_code = self.compute_f_code()
        self.df_code = self.compute_df_code()
        
        sympy.Basic.__str__ = lambda self: Printer().doprint(self)
        self.gradient_code = [str(self.gradient[i]) for i in range(3)]
        sympy.Basic.__str__ = old_ptr
        
        #print self.tracer_code
    
    # freeze template name
    def template_name(self): return 'ImplicitSurface'
    
    def f_code_template(self): return None
    
    def ia_poly(self, coeffs, n_coeff, var):
        s = []
        for i in range(n_coeff):
            if i == 0: c = '%s_0' % coeffs
            else:
                x = var;
                if i > 1: x = 'ia_pow%d(%s)' % (i,x)
                c = 'ia_mul_exact(%s,%s_%d)' % (x,coeffs,i)
            s.append(c)
        return ' + '.join(s);
        
    def compute_f_code(self):
        return """
            f = %s;""" % str(self.scaled_and_shifted)
    
    def compute_df_code(self):
        return """
            df = %s; """ % str(self.derivative)

    @staticmethod
    def print_pow(expr):
        "Print sympy pow expression in an OpenCL-friendly format"
        
        base = expr.args[0]
        exponent = expr.args[1]
        f = "pow"
        if exponent.is_integer:
            # pown was really slow on my gpu...
            n = int(exponent)
            if n > 0:
                base_str = "("+str(base)+")"
                return "("+('*'.join([base_str]*n))+")" 
            
        return "%s(%s,%s)" % (f, base, exponent)

