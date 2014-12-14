
# Tracer objects: Implicit surfaces

from tracer import *
from utils import normalize_tuple, vec_norm
import numpy
import sys
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


class QuaternionJuliaSet2(Tracer):
	
	def __init__(self, c, julia_itr, **kwargs):
		self.c = c
		self.julia_itr = julia_itr
		self.bndR = 1.5
		self.center = (0,0,0)
		self.max_itr = 100
		self.precision = 0.001
		self.max_step = 0.1
		
		# TODO: tuple(center)
		
		for (k,v) in kwargs.items(): setattr(self, k, v)
		
		self.tracer_code = """
		if (origin_self) return; // ------------- remember to remove this
		
		float trace_begin, trace_end;
		"""
		
		self.tracer_code += Sphere.get_bounding_volume_code(\
			self.center, self.bndR, 'trace_begin', 'trace_end');
		
		self.tracer_code += """
		
		int i=0;
		const int MAX_ITER = %d;
		const float TARGET_EPS = %s;
		const float MAX_STEP = %s;
		const float ESCAPE_RADIUS = 2.0f;
		""" % (self.max_itr, self.precision, self.max_step)
		
		self.tracer_code += """
		float step, dist;
		const float4 c = (float4)%s;
		dist = trace_begin;
		float3 pos = trace_begin * ray + origin - (float3)%s;
		
		""" % (self.c,self.center)
		
		self.tracer_code += """
		for( i=0; i < MAX_ITER; i++ )
		{
			float4 q = (float4)(pos,0), q1, qd = (float4)(1,0,0,0);
			
			float gr = ray.x, gi = ray.y, gj = ray.z, gk = 0,
				  gr1, gi1, gj1, gk1;
			
			for (int iii=0; iii<%d; ++iii)
			{
				if (origin_self)
				{
					// TODO: use quaternion derivative, if possible
				
					// Derivative chain rule
					gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
					gi1 = 2*(gr*q.y + q.x*gi);
					gj1 = 2*(gr*q.z + q.x*gj);
					gk1 = 2*(gr*q.w + q.x*gk);
					
					gr = gr1;
					gi = gi1;
					gj = gj1;
					gk = gk1;
				}
				
				qd = 2*quaternion_mult(q,qd);
				q = quaternion_square(q) + c;
				
			}
			
			float ql2 = dot(q,q);
			float ds = 2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk);
			
			// The magic distance estimate formula, see the 1989 article:
			// Hart, Sandin, Kauffman, "Ray Tracing Deterministic 3-D Fractals"
			float l = length(q);
			step = 0.5 * l * log(l) / length(qd);
			step = min(step, MAX_STEP);
			
			if (step < TARGET_EPS)
			{
				if (!origin_self || ds < 0)
				{ 
					*p_new_isec_dist = dist + step;
					return;
				}
				else
				{
					step = TARGET_EPS;
				}
			}
			/*if (ql2 < ESCAPE_RADIUS*ESCAPE_RADIUS)
			{
				*p_new_isec_dist = dist;
				return;
			}*/
			
			dist += step;
			
			if (dist > trace_end) return;
			pos += step * ray;
		}
		""" % self.julia_itr
		
		self.normal_code = """
		float4 q = (float4)(pos - (float3)%s,0);
		const float4 c = (float4)%s, q1;
		""" % (self.center,self.c)
		
		self.normal_code += """
		float3
			gr = (float3)(1,0,0),
			gi = (float3)(0,1,0),
			gj = (float3)(0,0,1),
			gk = (float3)(0,0,0),
			gr1, gi1, gj1, gk1;
		
		float qr1;
		
		for (int iii=0; iii<%d; ++iii)
		{
			// Derivative chain rule
			gr1 = 2*(q.x*gr - q.y*gi - q.z*gj - q.w*gk);
			gi1 = 2*(gr*q.y + q.x*gi);
			gj1 = 2*(gr*q.z + q.x*gj);
			gk1 = 2*(gr*q.w + q.x*gk);
			
			gr = gr1;
			gi = gi1;
			gj = gj1;
			gk = gk1;
			
			// Quaternion operation z -> z^2 + c
			q = quaternion_square(q) + c;
		}
		float3 grad = fast_normalize(2*(q.x*gr + q.y*gi + q.z*gj + q.w*gk));
		
		if (all(isfinite(grad))) *p_normal = grad;
		else *p_normal = (float3)(1,0,0);
		""" % self.julia_itr
	


class QuaternionJuliaSet(ImplicitSurface):
	
	def __init__(self, c, julia_itr, *args, **argd):
		self.c = c
		self.julia_itr = julia_itr
		ImplicitSurface.__init__(self, "x^2 + y^2 + z^2 - 1", *args, **argd)
		
		self.tracer_code = """
		if (origin_self) return;
		""" + self.tracer_code
	
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

