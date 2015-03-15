
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
                max_itr=1500, precision=0.001):
        
        self.unique_tracer_id = str(id(self))
        
        self.center = tuple(center)
        self.scale = scale
        
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
        
        self.tracer_code = """
        ia_type t;
        """
        
        self.tracer_code += Sphere.get_bounding_volume_code(\
            self.center, bndR*self.scale, 'ia_begin(t)', 'ia_end(t)');
        
        self.tracer_code += """
        
        int i=0;
        const int MAX_ITER = %d;
        const float TARGET_EPS = %s;
        const float FRACTION = 0.5;
        const float SELF_MAX_BEGIN_STEP = 0.1;
        """ % (max_itr, precision)
        
        self.tracer_code += """
        ia_type f, df;
        ia_type x,y,z;
        
        float step;
        int steps_since_subdiv = 0;
        
        const ia_type ray_x = ia_new_exact(ray.x);
        const ia_type ray_y = ia_new_exact(ray.y);
        const ia_type ray_z = ia_new_exact(ray.z);
        const ia_type origin_x = ia_new_exact(origin.x);
        const ia_type origin_y = ia_new_exact(origin.y);
        const ia_type origin_z = ia_new_exact(origin.z);
        int need_subdiv;
        
        if (origin_self)
        {
            ia_end(t) = min(ia_end(t),ia_begin(t) + SELF_MAX_BEGIN_STEP);
        }

        for( i=0; i < MAX_ITER; i++ )
        {
            if (ia_begin(t) >= old_isec_dist) return;
            if (ia_end(t) > old_isec_dist) ia_end(t) = old_isec_dist;
            
            x = ia_mul(ray_x,t) + origin_x;
            y = ia_mul(ray_y,t) + origin_y;
            z = ia_mul(ray_z,t) + origin_z;
        
            %s
            
            step = ia_len(t);
            need_subdiv = 0;
            
            if ((inside && ia_end(f) > 0) || (!inside && ia_begin(f) < 0))
            {
                need_subdiv = 1;
                
                if (origin_self)
                {
                    %s
                    
                    if (!ia_contains_zero(df)) {
                        if ( (ia_end(df) < 0) == inside ) need_subdiv = 0;
                    }
                }
                
                if ( need_subdiv )
                {
                    if (step < TARGET_EPS || i == MAX_ITER-1)
                    {
                        step = ia_center(t);
                        *p_new_isec_dist = step;
                        return;
                    }
                
                    // Subdivide
                    step *= FRACTION;
                    ia_end(t) = ia_begin(t) + step;
                    steps_since_subdiv = 0;
                    continue;
                }
                
            }
            steps_since_subdiv++;
            
            // Step forward
            if (steps_since_subdiv > 1) step /= FRACTION;
            t = ia_new(ia_end(t),ia_end(t)+step);
        }
        """ % (self.compute_f_code(), self.compute_df_code());
        
        sympy.Basic.__str__ = lambda self: Printer().doprint(self)
        
        self.normal_code = """
        *p_normal = fast_normalize((float3)(%s, %s, %s));
        """ % tuple([self.gradient[i] for i in range(3)])
        
        sympy.Basic.__str__ = old_ptr
        
        #print self.tracer_code
    
    # freeze template name
    def template_name(self): return 'ImplicitSurface'
    
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

