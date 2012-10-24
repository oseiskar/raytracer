
from utils import normalize
import numpy

class Tracer:
	TRACE_KERNEL_ARGUMENTS = """
	__global const float3 *p_origin,
	__global const float3 *p_ray,
	__global const float3 *p_last_normal,
	__global const float *p_old_isec_dist,
	__global float *p_new_isec_dist,
	__global const uint *p_inside,
	__global const uint *p_origin_self""" # TODO: bool
	
	NORMAL_KERNEL_ARGUMENTS = """
	__global const float3 *p_pos,
	__global float3 *p_normal
	"""
	
	TRACE_KERNEL_HEADER = """
	
	const int gid = get_global_id(0);
	const float3 origin = p_origin[gid];
	const float3 ray = p_ray[gid];
	const float3 last_normal = p_last_normal[gid];
	const float old_isec_dist = p_old_isec_dist[gid];
	const bool inside = p_inside[gid] != 0;
	const bool origin_self = p_origin_self[gid] != 0;
	p_new_isec_dist += gid;
	"""
	
	NORMAL_KERNEL_HEADER = """
	
	const int gid = get_global_id(0);
	const float3 pos = p_pos[gid];
	p_normal += gid;
	
	"""
	
	TRACE_STATIC_ARGUMENTS = """
	const float3 origin,
	const float3 ray,
	const float3 last_normal,
	const float old_isec_dist,
	__private float *p_new_isec_dist,
	bool inside,
	bool origin_self"""
	
	NORMAL_STATIC_ARGUMENTS = """
	const float3 pos,
	__global float3 *p_normal
	"""
	
	def make_kernel(self, kernel_function=True):
		
		if kernel_function:
			kernel_keyword = '__kernel'
			trace_arguments = Tracer.TRACE_KERNEL_ARGUMENTS
			normal_arguments = Tracer.NORMAL_KERNEL_ARGUMENTS
			trace_header = Tracer.TRACE_KERNEL_HEADER
			normal_header = Tracer.NORMAL_KERNEL_HEADER
		else:
			kernel_keyword = ''
			trace_arguments = Tracer.TRACE_STATIC_ARGUMENTS
			normal_arguments = Tracer.NORMAL_STATIC_ARGUMENTS
			trace_header = ''
			normal_header = ''
		
		kernel_id = self.__class__.__name__+str(id(self))
		self.tracer_kernel_name = kernel_id+"_tracer"
		self.normal_kernel_name = kernel_id+"_normal"
		
		self.kernel_code = kernel_keyword + " void %s(%s) {" \
			% (self.tracer_kernel_name, trace_arguments)
		self.kernel_code += trace_header
		self.kernel_code += "\n" + self.tracer_code + "\n}\n\n"
		
		self.kernel_code += kernel_keyword + " void %s(%s) {" \
			% (self.normal_kernel_name, normal_arguments)
		self.kernel_code += normal_header
		self.kernel_code += "\n" + self.normal_code + "\n}\n\n"
		
		return self.kernel_code
		

class Sphere(Tracer):
	
	def _params(self): return """
	const float3 center = (float3)%s;
	const float R2 = %s;
	const float invR = %s;
	""" % (self.pos, self.R**2, 1.0/self.R)
	
	@property
	def tracer_code(self): return self._params() + \
	"""
	float3 rel = center - origin;
	float dotp = dot(ray, rel);
	float psq = dot(rel, rel);
	
	float dist, discr, sqrdiscr;
	
	if (origin_self && !inside)
	{
		return;
		// no intersection
	}
	
	if (dotp <= 0 && !inside)
	{
		// no intersection
		return;
	}
	
	discr = dotp*dotp - psq + R2;
	if(discr < 0) return;
	
	sqrdiscr = native_sqrt(discr);
	dist = dotp - sqrdiscr;
	
	if (inside) dist = dotp + sqrdiscr;
	else dist = dotp - sqrdiscr;
	
	if (dist <= 0) return;
	*p_new_isec_dist = dist;
	"""
	
	@property
	def normal_code(self): return self._params() + """
	*p_normal = (pos - center) * invR;
	"""
	
	def __init__(self, pos, R):
		self.pos = pos
		self.R = R

class ImplicitSurface(Tracer):
	# The equation defining the surface must be positive outside the object
	# (multiply eq. by -1 if things do not work)
	
	def __init__(self, eq,
				center=(0,0,0), scale=1.0, bndR=None,
				max_itr=1000, precision=0.001):
		
		self.center = center
		self.scale = scale
		
		import sympy
		import sympy.core.numbers
		
		x,y,z = sympy.symbols('x y z')
		
		self.eq = sympy.sympify(eq).subs([\
			(x,((x-self.center[0])/self.scale)),
			(y,((y-self.center[1])/self.scale)),
			(z,((z-self.center[2])/self.scale))])
		
		self.gx = sympy.diff(self.eq,x)
		self.gy = sympy.diff(self.eq,y)
		self.gz = sympy.diff(self.eq,z)
		
		#print eq
		#print gx
		#print gy
		#print gz
		
		# Must replace some expressions to make it OpenCL
		class Printer(sympy.printing.str.StrPrinter):
			def _print_Pow(self, expr):
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
				
		
		old_ptr = sympy.Basic.__str__
		sympy.Basic.__str__ = lambda self: IAPrinter().doprint(self)
		
		if bndR:
			bndR *= self.scale
		
			self.tracer_code = """
			// Bounding sphere intersection
			const float R2 = %s;
			const float3 center = (float3)%s;
			float3 rel = center - origin;
			float dotp = dot(ray, rel);
			float psq = dot(rel, rel);
			
			bool inside_bnd = psq < R2;
			
			if (dotp <= 0 && !inside_bnd)
			{
				// no intersection
				return;
			}
			
			const float discr = dotp*dotp - psq + R2;
			if(discr < 0) return;
			const float sqrdiscr = native_sqrt(discr);
			
			ia_type cur_ival = ia_new(dotp - sqrdiscr, dotp + sqrdiscr);
			ia_end(cur_ival) = min(ia_end(cur_ival),old_isec_dist);
			ia_begin(cur_ival) = max(ia_begin(cur_ival),0.0f);
			if (ia_end(cur_ival) <= ia_begin(cur_ival)) return;
			
			""" % (bndR**2, tuple(self.center))
		else:
			self.tracer_code = """
			ia_type cur_ival = ia_new(0,old_isec_dist);
			"""
		
		self.tracer_code += """
		
		int i=0;
		const int MAX_ITER = %d;
		const float TARGET_EPS = %s;
		const float FRACTION = 0.5;
		const float SELF_MAX_BEGIN_STEP = 0.01;
		""" % (max_itr, precision)
		
		self.tracer_code += """
		ia_type x, y, z, f, df;
		
		float step;
		int steps_since_subdiv = 0;
		
		if (origin_self)
		{
			ia_end(cur_ival) = SELF_MAX_BEGIN_STEP;
		}

		for( i=0; i < MAX_ITER; i++ )
		{
			if (ia_begin(cur_ival) >= old_isec_dist) return;
			if (ia_end(cur_ival) > old_isec_dist) ia_end(cur_ival) = old_isec_dist;
			
			x = ia_add_exact(ia_mul_exact(cur_ival, ray.x), origin.x);
			y = ia_add_exact(ia_mul_exact(cur_ival, ray.y), origin.y);
			z = ia_add_exact(ia_mul_exact(cur_ival, ray.z), origin.z);
			
			%s
			
			step = ia_len(cur_ival);
			
			if (ia_contains_zero(f))
			//if (ia_begin(f) < 0)
			{
				if (step < TARGET_EPS || i == MAX_ITER-1)
				{
					step = ia_center(cur_ival);
					*p_new_isec_dist = step;
					return;
				}
				
				if (origin_self)
				{
					%s
				}
				
				if ( !origin_self || (ia_begin(df)<0) != inside )
				{
					// Subdivide
					step *= FRACTION;
					ia_end(cur_ival) = ia_begin(cur_ival) + step;
					steps_since_subdiv = 0;
					continue;
				}
			}
			steps_since_subdiv++;
			
			// Step forward
			if (steps_since_subdiv > 1) step /= FRACTION;
			cur_ival = ia_new(ia_end(cur_ival),ia_end(cur_ival)+step);
		}
		""" % (self.compute_f_code(), self.compute_df_code());
		
		#print eq
		#print gx
		#print gy
		#print gz
		
		sympy.Basic.__str__ = lambda self: Printer().doprint(self)
		
		x.name = 'pos.x'
		y.name = 'pos.y'
		z.name = 'pos.z'
		self.normal_code = self.compute_normal_code()
		
		sympy.Basic.__str__ = old_ptr
		
		#print self.tracer_code
		
	def compute_f_code(self):
		return "f = %s;" % self.eq
	
	def compute_df_code(self):
		return """
			df = ia_add(
				ia_add(
					ia_mul_exact(%s,ray.x),
					ia_mul_exact(%s,ray.y)),
				ia_mul_exact(%s,ray.z));
		""" % (self.gx,self.gy,self.gz)
	
	def compute_normal_code(self):
		return """
		*p_normal = fast_normalize((float3)(%s, %s, %s));
		""" % (self.gx,self.gy,self.gz)

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
	
	def compute_df_code(self):
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
			
		/*float drdx=1, drdy=0, drdz=0,
		      didx=0, didy=1, didz=0,
		      djdx=0, djdy=0, djdz=1,
		      dkdx=0, dkdy=0, dkdz=0;
		
		float drdx1, drdy1, drdz1,
		      didx1, didy1, didz1,
		      djdx1, djdy1, djdz1,
		      dkdx1, dkdy1, dkdz1;*/
		
		const float cr = %s, ci = %s, cj = %s, ck = %s;
		float qr1;
		
		for (int iii=0; iii<%d; ++iii)
		{
			// Derivative chain rule...
			
			/*drdx1 = 2*(drdx*qr - didx*qi - djdx*qj - dkdx*qk);
			drdy1 = 2*(drdy*qr - didy*qi - djdy*qj - dkdy*qk);
			drdz1 = 2*(drdz*qr - didz*qi - djdz*qj - dkdz*qk);
			
			didx1 = 2*(drdx*qi + didx*qr);
			didy1 = 2*(drdy*qi + didy*qr);
			didy1 = 2*(drdz*qi + didz*qr);
			
			djdx1 = 2*(drdx*qj + djdx*qr);
			djdy1 = 2*(drdy*qj + djdy*qr);
			djdy1 = 2*(drdz*qj + djdz*qr);
		
			dkdx1 = 2*(drdx*qk + dkdx*qr);
			dkdy1 = 2*(drdy*qk + dkdy*qr);
			dkdy1 = 2*(drdz*qk + dkdz*qr);*/
			
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
			
			/*drdx=drdx1; drdy=drdy1; drdz=drdz1;
			didx=didx1; didy=didy1; didz=didz1;
			djdx=djdx1; djdy=djdy1; djdz=djdz1;
			dkdx=dkdx1; dkdy=dkdy1; dkdz=dkdz1;*/
		}
		
		/*float dx = 2*(qr*drdx + qi*didx + qj*djdx + qk*dkdx);
		float dy = 2*(qr*drdy + qi*didy + qj*djdy + qk*dkdy);
		float dz = 2*(qr*drdz + qi*didz + qj*djdz + qk*dkdz);*/
		
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
		

class HalfSpace(Tracer):
		
	def _params(self): return """
	const float3 normal = (float3)%s;
	const float h = %s;
	""" % (self.normal_vec, self.h)
	
	@property
	def tracer_code(self): return self._params() + """
	
	if (!origin_self)
	{
		float slope = dot(ray,-normal);
		float dist = dot(origin, normal)+h;
		
		dist = dist/slope;
		if (dist > 0) *p_new_isec_dist = dist;
	}
	"""
	
	@property
	def normal_code(self): return self._params() + """
	*p_normal = normal;
	"""
	
	def __init__(self, normal, h):
		self.normal_vec = tuple(normalize(numpy.array(normal)))
		self.h = h
