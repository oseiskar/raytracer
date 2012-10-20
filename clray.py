import numpy as np
import pyopencl.clrandom as cl_random
import pyopencl.clmath as cl_math
import pyopencl as cl
import time

from accelerator import Accelerator
from objects import *

startup_time = time.time()

# ------- General options

use_pygame = True
use_scipy_misc_pil_image = True
output_raw_data = True

brightness = 0.5
quasirandom = False
interactive_opencl_context_selection = False
samples_per_pixel = 200256
min_bounces = 2
russian_roulette_prob = .3
#russian_roulette_prob = -1
max_bounces = 7

#imgdim = (640,400)
imgdim = (800,600)
#imgdim = (1024,768)


# ------- Image output

pgwin = None
def show_and_save_image( imgdata ):
	
	if output_raw_data:
		np.save('out.raw.npy', imgdata)
	
	ref = np.mean(imgdata)
	imgdata = (np.clip(imgdata/ref*brightness, 0, 1)*255).astype(np.uint8)
	
	if use_pygame:
		import pygame
		h,w = imgdata.shape[:2]
		global pgwin
		if not pgwin:
			pgwin = pygame.display.set_mode((w,h))
			pygame.display.set_caption("Raytracer") 
		
		pgwin.blit(pygame.surfarray.make_surface(imgdata.transpose((1,0,2))), (0,0))
		pygame.display.update()
	
	if use_scipy_misc_pil_image:
		from scipy.misc import toimage
		toimage(imgdata).save('out.png')
	

# ------- Numpy utils

def normalize(vecs):
	lens = np.sum(vecs**2, len(vecs.shape)-1)
	lens = np.sqrt(lens)
	lens = np.array(lens)
	lens.shape += (1,)
	return vecs / lens
	
def rotmat_tilt_camera(xa, ya):
	
	rm2d = lambda a : np.array([[np.cos(a), -np.sin(a)],[np.sin(a),np.cos(a)]])
	
	
	rot3dy = np.identity(3)
	rot3dy[1:,1:] = rm2d(ya)
	
	rot3dx = np.identity(3)
	
	rot3dx[[[0],[2]],[0,2]] = rm2d(xa)
	
	return np.dot(rot3dy, rot3dx)

def camera_rotmat(direction, up=(0,0,1)):
	
	direction = np.array(direction)
	up = np.array(up)
	
	right = np.cross(direction,up)
	up = np.cross(right,direction)
	
	rotmat = np.vstack((right,-up,direction))
	
	return normalize(rotmat).transpose()
	
def camera(wh, flat=False, wfov=60, direction=(0,1,0), up=(0,0,1)):
	
	w,h = wh
	wfov = wfov/180.0*np.pi
	aspect = h / float(w)
	hfov = wfov * aspect
	
	rotmat = camera_rotmat(direction, up)
	
	#tilt = rotmat_tilt_camera(0.3,0.4)
	#rotmat = np.dot(rotmat, tilt)
	
	if flat:
		ra = np.tan(wfov * 0.5)
		xr = np.linspace(-ra,ra,w)
		yr = np.linspace(-ra*aspect,ra*aspect,h)
		X,Y = np.meshgrid(xr,yr)
		Z = np.ones(X.shape)
	else:
		pixel_angle = float(wfov)/w;
		xa = (np.arange(0,w)+0.5)*pixel_angle - wfov/2.0
		ya = (np.arange(0,h)+0.5)*pixel_angle - hfov/2.0
		Xa,Ya = np.meshgrid(xa,ya)
		
		X = np.sin(Xa)*np.cos(Ya)
		Z = np.cos(Xa)*np.cos(Ya)
		Y = np.sin(Ya)
	
	N = w*h
	vecs = np.dstack((X,Y,Z))
	vecs = np.reshape(vecs, (N,3)).transpose()
	vecs = np.dot(rotmat, vecs).transpose()
	vecs = np.reshape(vecs, (h,w,3))
	
	if flat: vecs = normalize(vecs)
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
	
	if hemisphere: n = n*2
	
	golden_angle = np.pi * (3 - np.sqrt(5))
	theta = golden_angle * np.arange(n)
	z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
	radius = np.sqrt(1 - z * z)
	
	points = np.zeros((n, 3))
	points[:,0] = radius * np.cos(theta)
	points[:,1] = radius * np.sin(theta)
	points[:,2] = z
	
	if hemisphere:
		n = n/2;
		points = points[:n,:]
	
	return points

# ------------- Define scene

def make_world_box( dims, center=(0,0,0) ):
	return [\
		HalfSpace( ( 1, 0, 0), dims[0]-center[0] ), \
		HalfSpace( (-1, 0, 0), dims[0]+center[0] ), \
		HalfSpace( ( 0, 1, 0), dims[1]-center[1] ), \
		HalfSpace( ( 0,-1, 0), dims[1]+center[1] ), \
		HalfSpace( ( 0, 0, 1), dims[2]-center[2] ), \
		HalfSpace( ( 0, 0,-1), dims[2]+center[2] )]



objects = []
objects += make_world_box( (3,5,2), (0,0,2) );

#sphere = Sphere( (0,2,1.0), 1.0 )
#objects.append(sphere)

objects.append(HalfSpace( tuple(normalize(np.array((-1,-1,-2)))), 5 ))

#equation='x**2 + y**2 + z**2 - 1.0**2'
equation='x**4 - 5*x**2 + y**4 - 5*y**2 + z**4 - 5*z**2 + 11.8'
objects.append(ImplicitSurface((0.2,1.5,1.0),equation, 0.5))

light = Sphere( (-3,-1,2), 0.5 )
objects.append(light)

Nobjects = len(objects)
object_materials = Nobjects*['white']
object_materials[4] = 'green'
#object_materials[5] = 'sky'

object_materials[-3] = 'sky'
object_materials[-2] = 'mirror'
object_materials[-1] = 'light'

materials = {\
'default': # "Air" / initial / default material
	{ 'diffuse': ( 1, 1, 1),
	  'emission':(0, 0, 0),
	  'reflection':(0,0,0),
	  'transparency': (0,0,0),
	  'ior': (1.0,), # Index Of Refraction
	  'vs': (0,0,0) #(0.1,0.1,0.1) # "fog"
	}, 
	# --- Other materials
'white':
	{ 'diffuse': ( .8, .8, .8) }, 
'mirror':
	{ 'diffuse': (.2,.2,.2), 'reflection':(.7,.7,.7) },
'red':
	{ 'diffuse': (.7,.4,.4) }, 
'light':
	{ 'diffuse': ( 1, 1, 1), 'emission':(4,2,.7) },
'sky':
	{ 'diffuse': ( 0, 0, 0), 'emission':tuple(np.array((.5,.5,.7))*0.7) },
'glass':
	{ 'diffuse': (.1,.1,.1), 'transparency':(.4,.7,.4), 'reflection':(.2,.2,.2), 'ior':(1.5,)},
'wax':
	{ 'diffuse': (0.3,0.5,0), 'reflection': (.2,.2,.0), 'transparency':(1.,1.,1.), 'vs':(.02,.04,.02), 'ior':(1.02,)},
'green':
	{ 'diffuse': (0.4,0.9,0.4)}
}

camera_target = np.array((0,2,0.4))
camera_pos = np.array((2,-3,3.5))
camera_fov = 60
camera_dir = camera_target - camera_pos
flat_camera = False
cam = camera(imgdim, flat_camera, camera_fov, camera_dir)

#fovx = np.tan(camera_fov*0.5 / 180 * np.pi)
#fovy = fovx * float(imgdim[1])/imgdim[0]
rotmat = camera_rotmat(camera_dir)
fovx_rad = camera_fov / 180.0 * np.pi
pixel_angle = fovx_rad / imgdim[0]

# ------------- make OpenCL code

kernels = set([obj.make_kernel(False) for obj in objects])

utils = open('utils.cl', 'r').read() # static code


# ------------- make tracer kernel (finds intersections)
trace_kernel = """
__kernel void trace(
	__global float3 *p_pos,
	__global const float3 *p_ray,
	__global float3 *p_normal,
	__global float *p_isec_dist,
	__global uint *p_whichobject,
	__global const uint *p_inside)
{

	const int gid = get_global_id(0);
	const float3 ray = p_ray[gid];
	float3 pos = p_pos[gid];
	const float3 last_normal = p_normal[gid];
	const uint lastwhichobject = p_whichobject[gid];
	const uint inside = p_inside[gid];
	
	p_whichobject += gid;
	p_normal += gid;
	p_isec_dist += gid;
	p_pos += gid;
	
	float old_isec_dist = *p_isec_dist;
	float new_isec_dist = 0;
	
	uint i = 0;
	uint whichobject = 0;
"""

# Unroll loop to CL code
for i in range(len(objects)):
	
	obj = objects[i]
	
	tracer_name = obj.tracer_kernel_name
	
	trace_kernel += """
	new_isec_dist = 0;
	i = %s;
	""" % (i+1)
	
	trace_kernel += """
	// call tracer
	%s(pos, ray, last_normal, old_isec_dist, &new_isec_dist, inside == i, lastwhichobject == i);
	""" % tracer_name
	
	# TODO: handle non-hitting rays!
	
	trace_kernel += """
	if (//lastwhichobject != i && // cull self
	    new_isec_dist > 0 &&
	    new_isec_dist < old_isec_dist)
	{
		old_isec_dist = new_isec_dist;
		whichobject = i;
	}
	"""

trace_kernel += """
	pos += old_isec_dist * ray; // saxpy
"""

for i in range(len(objects)):
	
	obj = objects[i]
	normal_name = obj.normal_kernel_name
	
	trace_kernel += """
	i = %s;
	""" % (i+1)
	
	trace_kernel += """
	if (whichobject == i)
	{
		// call normal
		%s(pos, p_normal);
		if (inside == i) *p_normal = -*p_normal;
	}
	""" % obj.normal_kernel_name

trace_kernel += """
	*p_isec_dist = old_isec_dist;
	*p_whichobject = whichobject;
	*p_pos = pos;
}
"""

# ------------- shader kernel

shader_kernel_params = """
#define RUSSIAN_ROULETTE_PROB %s
""" % russian_roulette_prob

shader_kernel = shader_kernel_params + open('shader.cl', 'r').read()

prog_code = utils
prog_code += "\n".join(list(kernels))
prog_code += trace_kernel
prog_code += shader_kernel

cur_code_file = open('last_code.cl', 'w')
cur_code_file.write(prog_code)
cur_code_file.close()


# ------------- Initialize CL

acc = Accelerator(interactive_opencl_context_selection)
prog = cl.Program(acc.ctx, prog_code).build()

# Utils

profiling_info = {}

def output_profiling_info():
	total = 0
	tatotal = 0
	for (k,v) in profiling_info.items():
		t = v['t']*1e-9
		ta = v['ta']
		n = v['n']
		fmt = '%.2g'
		print ('%d\t'+('\t'.join([fmt]*4))+'\t'+k) % (n,t,ta,t/n,ta/n)
		total += t
		tatotal += ta
	print '----', total,'or',tatotal, 'seconds total'

def prog_caller(N):
	def prog_call(kernel_name, buffer_args, value_args=tuple([])):
		
		t1 = time.time()
		#acc.queue.finish()
		kernel = getattr(prog,kernel_name)
		arg =  tuple([x.data for x in buffer_args]) + value_args
		event = kernel(acc.queue, (N,), None, *arg)
		event.wait()
		
		t = (event.profile.end - event.profile.start)
		if kernel_name not in profiling_info:
			profiling_info[kernel_name] = {'n':0, 't':0, 'ta':0}
			
		profiling_info[kernel_name]['t'] += t
		profiling_info[kernel_name]['n'] += 1
		profiling_info[kernel_name]['ta'] += time.time() - t1
		#print kernel_name, "executed in", t*1e-9, "s"
		
	return prog_call


# ------------- Parameter arrays

# Materials
fog = False

def new_const_buffer(buf):
	mf = cl.mem_flags
	return cl.Buffer(acc.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=buf.astype(np.float32))
	
def new_mat_buf(pname):
	default = materials['default'][pname]
	if len(default) > 1:
		w = 4
		default += (0,)
	else:
		w = 1
		
	buf = np.zeros((Nobjects+1,w))
	buf[0] = np.array(default)
	for i in range(Nobjects):
		if pname in materials[object_materials[i]]:
			prop = materials[object_materials[i]][pname]
			if w > 1: prop += (0,)
		else:
			prop = default
		buf[i+1] = np.array(prop)
	
	if pname == 'vs' and buf.sum() != 0:
		fog = True
		print "fog"
	return new_const_buffer(buf)

mat_diffuse = new_mat_buf('diffuse')
mat_emission = new_mat_buf('emission')
mat_reflection = new_mat_buf('reflection')
mat_transparency = new_mat_buf('transparency')
mat_vs = new_mat_buf('vs')
mat_ior = new_mat_buf('ior')
max_broadcast_vecs = 4
vec_broadcast = new_const_buffer(np.zeros((max_broadcast_vecs,4)))

# ---- Path tracing

# Params
cam = acc.make_vec3_array(cam)
N = cam.size / 4
itr_per_refresh = 10
luxury = 2
imgshape = imgdim[::-1]
caching = False

prog_call = prog_caller(N)
def memcpy(dst,src): cl.enqueue_copy(acc.queue, dst.data, src.data)
def fill_vec(data, vec):
	#prog_call('fill_vec', (data,), tuple(np.float32(vec)))
	hostbuf = vec.astype(np.float32)
	cl.enqueue_copy(acc.queue, vec_broadcast, hostbuf)
	prog_call('fill_vec_broadcast', (data,), (vec_broadcast,))

# Randomization init
qdirs = quasi_random_direction_sample(samples_per_pixel)
qdirs = np.random.permutation(qdirs)

rnd = cl_random.RanluxGenerator(acc.queue, N, luxury)

# Device buffers. 
img = acc.new_vec3_array(imgshape)

whichobject = acc.new_array(imgshape, np.uint32, True)
pos = acc.zeros_like(cam)
ray = acc.zeros_like(pos)

inside = acc.zeros_like(whichobject)

normal = acc.zeros_like(pos)
isec_dist = acc.zeros_like(img)

raycolor = acc.zeros_like(img)
curcolor = acc.zeros_like(raycolor)

firstray = acc.zeros_like(pos)
firstpos = acc.zeros_like(pos)
firstwhichobject = acc.zeros_like(whichobject)
firstnormal = acc.zeros_like(normal)
firstraycolor = acc.zeros_like(raycolor)
directlight = acc.zeros_like(img)

# Do it
for j in xrange(samples_per_pixel):
	
	t0 = time.time()
	
	if j==0 or not caching:
		
		if caching: memcpy(ray, cam)
		else:
			# TODO: quasi random...
			sx = np.float32(np.random.rand())
			sy = np.float32(np.random.rand())
			
			overlap = 0.0
			thetax = (sx-0.5)*pixel_angle*(1.0+overlap)
			thetay = (sy-0.5)*pixel_angle*(1.0+overlap)
			
			tilt = rotmat_tilt_camera(thetax,thetay)
			mat = np.dot(np.dot(rotmat,tilt),rotmat.transpose())
			mat4 = np.zeros((4,4))
			mat4[0:3,0:3] = mat
			
			cl.enqueue_copy(acc.queue, vec_broadcast,  mat4.astype(np.float32))
			prog_call('subsample_transform_camera', (cam,ray,), (vec_broadcast,))
		
		fill_vec(pos, camera_pos)
		whichobject.fill(0)
		normal.fill(0)
		raycolor.fill(1)
		curcolor.fill(0)
		kbegin = 0
	else:
		# cached first intersection
		memcpy(pos, firstpos)
		memcpy(whichobject, firstwhichobject)
		memcpy(normal, firstnormal)
		memcpy(raycolor, firstraycolor)
		memcpy(ray, firstray)
		kbegin = 1
		# TODO: firstinside
	
	inside.fill(0)
	isec_dist.fill(0) # TODO
	
	k = kbegin
	while True:
	#for k in xrange(kbegin,min_bounces+1):
		
		#if k == min_bounces+1: break
		r_prob = 1
		if k > min_bounces:
			rand_01 = np.random.rand()
			if rand_01 < russian_roulette_prob and k < max_bounces:
				r_prob = 1.0/(1-russian_roulette_prob)
			else:
				break
				
		#inside.fill(0)
		
		vec = (0,0,0) # dummy
		if k!=0:
			if quasirandom and k == 1:
				vec = qdirs[j,:]
			else:
				vec = normalize(np.random.normal(0,1,(3,)))
			
		vec = np.array(vec).astype(np.float32) 
		rand_01 = np.float32(np.random.rand())
		
		if k != 0:
			hostbuf = np.zeros((3,4), dtype=np.float32)
			hostbuf[0,:3] = vec
			#hostbuf[1,:3] = light.pos
			#hostbuf[2,0] = light.R
			cl.enqueue_copy(acc.queue, vec_broadcast, hostbuf)
			prog_call('prob_select_ray', \
				(img,whichobject, normal,isec_dist,pos,ray,raycolor,inside), \
				(mat_emission, mat_diffuse,mat_reflection,mat_transparency,mat_ior,mat_vs,\
				rand_01,vec_broadcast))
				
		raycolor *= r_prob
		
		memcpy(curcolor, raycolor)
		
		isec_dist.fill(100) # TODO
		prog_call('trace', (pos,ray,normal,isec_dist,whichobject,inside))
		
		if j==0 and k==0 and caching:
			# cache first intersection
			memcpy(firstpos, pos)
			memcpy(firstwhichobject,whichobject)
			memcpy(firstnormal,normal)
			memcpy(firstraycolor,raycolor)
			memcpy(firstray, ray)
			memcpy(directlight,img)
			img.fill(0)
			
		k += 1
	
	img += directlight
	
	if not fog:
		prog_call('mult_by_param_vec_vec', (whichobject, curcolor), (mat_emission,))
		img += curcolor
	
	tcur = time.time()
	print '%d/%d'%(j+1,samples_per_pixel),"time per image:", (tcur-t0), "total:", (tcur-startup_time), "k=%d"%k
	
	if j % itr_per_refresh == 0 or j==samples_per_pixel-1:
		show_and_save_image( img.get().astype(np.float32)[...,0:3] )
		output_profiling_info()
		
			
output_profiling_info()
