
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
		
		self.tracer_code += ImplicitSurface.get_bounding_volume_code(\
			tuple(self.center), self.bndR, 'trace_begin', 'trace_end');
		
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
		
		""" % (tuple(self.c),tuple(self.center))
		
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
		""" % (tuple(self.center),tuple(self.c))
		
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
