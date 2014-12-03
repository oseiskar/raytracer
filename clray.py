import numpy as np
import time, sys, os, os.path
#import objgraph

from accelerator import Accelerator
from utils import *
from imgutils import Image

startup_time = time.time()

# ------- General options

old_raw_file = None
png_output_file = 'out.png'
raw_output_file = 'out.raw.npy'

interactive_opencl_context_selection = False

itr_per_refresh = 100

# ------- Import scene
sys.path.append('scenes/')
scene_filename = "scene-dev"
if len(sys.argv) > 1: scene_filename = sys.argv[1]

scene_module = __import__(scene_filename)
scene = scene_module.scene

# ------------- Initialize image

if len(sys.argv) > 2: old_raw_file = sys.argv[2]
image = Image( old_raw_file )
image.gamma = scene.gamma
image.brightness = scene.brightness

# ------------- set up camera

cam = scene.get_camera_rays()
rotmat = scene.get_camera_rotmat()
fovx_rad = scene.camera_fov / 180.0 * np.pi
pixel_angle = fovx_rad / scene.image_size[0]

# ------------- Find root container object
Nobjects = len(scene.objects)
root_object_id = 0
for i in range(Nobjects):
	if scene.root_object == scene.objects[i]:
		root_object_id = i+1

# ------------- make OpenCL code

kernel_map = {}
for obj in scene.objects:
	for (k,v) in obj.tracer.make_functions().items():
		if k in kernel_map and kernel_map[k] != v:
			raise "kernel name clash!!"
		kernel_map[k] = v
kernels = set(kernel_map.values())

objects = [obj.tracer for obj in scene.objects]
object_materials = [obj.material for obj in scene.objects]

cl_utils = open('utils.cl', 'r').read() # static code



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
	uint subobject;
	uint cur_subobject;
	
	uint i = 0;
	uint whichobject = 0;
"""

# Unroll loop to CL code
for i in range(Nobjects):
	
	obj = objects[i]
	
	trace_kernel += """
	new_isec_dist = 0;
	i = %s;
	
	// call tracer
	""" % (i+1)
	
	trace_kernel += obj.make_tracer_call([ \
			"pos",
			"ray",
			"last_normal",
			"old_isec_dist",
			"&new_isec_dist",
			"&cur_subobject",
			"inside == i",
			"lastwhichobject == i"])
	
	# TODO: handle non-hitting rays!
	
	trace_kernel += """
	if (//lastwhichobject != i && // cull self
	    new_isec_dist > 0 &&
	    new_isec_dist < old_isec_dist)
	{
		old_isec_dist = new_isec_dist;
		whichobject = i;
		subobject = cur_subobject;
	}
	"""

trace_kernel += """
	pos += old_isec_dist * ray; // saxpy
"""

for i in range(Nobjects):
	
	obj = objects[i]
	
	trace_kernel += """
	i = %s;
	""" % (i+1)
	
	trace_kernel += """
	if (whichobject == i)
	{
		// call normal
		%s
		if (inside == i) *p_normal = -*p_normal;
	}
	""" % obj.make_normal_call(["pos", "subobject", "p_normal"])

trace_kernel += """
	*p_isec_dist = old_isec_dist;
	*p_whichobject = whichobject;
	*p_pos = pos;
}
"""

# ------------- shader kernel

shader_kernel_params = """
#define RUSSIAN_ROULETTE_PROB %s
""" % scene.russian_roulette_prob

shader_kernel = shader_kernel_params + open('shader.cl', 'r').read()

prog_code = cl_utils

for kernel in kernels:
	curl = kernel.find('{')
	declaration = kernel[:curl] + ';\n'
	prog_code += declaration

prog_code += "\n"

prog_code += "\n".join(list(kernels))
prog_code += trace_kernel
prog_code += shader_kernel

cur_code_file = open('last_code.cl', 'w')
cur_code_file.write(prog_code)
cur_code_file.close()

# ------------- Initialize CL

acc = Accelerator(cam.size / 3, interactive_opencl_context_selection)
prog = acc.build_program( prog_code )

# ------------- Parameter arrays

# Materials
fog = False

def new_mat_buf(pname):
	default = scene.materials['default'][pname]
	if len(default) > 1:
		w = 4
		default += (0,)
	else:
		w = 1
		
	buf = np.zeros((Nobjects+1,w))
	buf[0] = np.array(default)
	for i in range(Nobjects):
		if pname in scene.materials[object_materials[i]]:
			prop = scene.materials[object_materials[i]][pname]
			if w > 1: prop += (0,)
		else:
			prop = default
		buf[i+1] = np.array(prop)
	
	if pname == 'vs' and buf.sum() != 0:
		fog = True
		print "fog"
	return acc.new_const_buffer(buf)

mat_diffuse = new_mat_buf('diffuse')
mat_emission = new_mat_buf('emission')
mat_reflection = new_mat_buf('reflection')
mat_transparency = new_mat_buf('transparency')
mat_vs = new_mat_buf('vs')
mat_ior = new_mat_buf('ior')
max_broadcast_vecs = 4
vec_broadcast = acc.new_const_buffer(np.zeros((max_broadcast_vecs,4)))


# ---- Path tracing

cam = acc.make_vec3_array(cam)
imgshape = scene.image_size[::-1]

def memcpy(dst,src): acc.enqueue_copy(dst.data, src.data)
def fill_vec(data, vec):
	hostbuf = np.float32(vec)
	acc.enqueue_copy(vec_broadcast, hostbuf)
	acc.call('fill_vec_broadcast', (data,), (vec_broadcast,))

# Randomization init
qdirs = quasi_random_direction_sample(scene.samples_per_pixel)
qdirs = np.random.permutation(qdirs)

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
directlight = acc.zeros_like(img)

# Do it
for j in xrange(scene.samples_per_pixel):
	
	t0 = time.time()
	
	cam_origin = scene.camera_position
	
	# TODO: quasi random...
	sx = np.float32(np.random.rand())
	sy = np.float32(np.random.rand())
	
	# Tent filter as in smallpt
	if scene.tent_filter:
		def tent_filter_transformation(x):
			x *= 2
			if x < 1: return np.sqrt(x)-1
			else: return 1-np.sqrt(2-x)
		
		sx = tent_filter_transformation(sx)
		sy = tent_filter_transformation(sy)
	
	overlap = 0.0
	thetax = (sx-0.5)*pixel_angle*(1.0+overlap)
	thetay = (sy-0.5)*pixel_angle*(1.0+overlap)
	
	dofx, dofy = random_dof_sample()
	
	dof_pos = (dofx * rotmat[:,0] + dofy * rotmat[:,1]) * scene.camera_dof_fstop
	
	sharp_distance = scene.camera_sharp_distance
	
	tilt = rotmat_tilt_camera(thetax,thetay)
	mat = np.dot(np.dot(rotmat,tilt),rotmat.transpose())
	mat4 = np.zeros((4,4))
	mat4[0:3,0:3] = mat
	mat4[3,0:3] = dof_pos
	mat4[3,3] = sharp_distance
	
	cam_origin = cam_origin + dof_pos
	
	acc.enqueue_copy(vec_broadcast,  mat4.astype(np.float32))
	acc.call('subsample_transform_camera', (cam,ray,), (vec_broadcast,))
	
		
	fill_vec(pos, cam_origin)
	whichobject.fill(0)
	normal.fill(0)
	raycolor.fill(1)
	curcolor.fill(0)
	kbegin = 0
	
	inside.fill(root_object_id)
	isec_dist.fill(0) # TODO
	
	k = kbegin
	r_prob = 1
	while True:
		
		#inside.fill(0)
		
		raycolor *= r_prob
		
		#memcpy(curcolor, raycolor)
		
		isec_dist.fill(scene.max_ray_length)
		acc.call('trace', (pos,ray,normal,isec_dist,whichobject,inside))
		
		if scene.quasirandom and k == 1:
			vec = qdirs[j,:]
		else:
			vec = normalize(np.random.normal(0,1,(3,)))
			
		vec = np.array(vec).astype(np.float32) 
		rand_01 = np.float32(np.random.rand())
		
		hostbuf = np.zeros((3,4), dtype=np.float32)
		hostbuf[0,:3] = vec
		#hostbuf[1,:3] = light.pos
		#hostbuf[2,0] = light.R
		acc.enqueue_copy(vec_broadcast, hostbuf)
		acc.call('prob_select_ray', \
			(img, whichobject, normal,isec_dist,pos,ray,raycolor,inside), \
			(mat_emission, mat_diffuse,mat_reflection,mat_transparency,mat_ior,mat_vs,\
			rand_01,vec_broadcast))
		
		r_prob = 1
		if k >= scene.min_bounces:
			rand_01 = np.random.rand()
			if rand_01 < scene.russian_roulette_prob and k < scene.max_bounces:
				r_prob = 1.0/(1-scene.russian_roulette_prob)
			else:
				break
		
		k += 1
	
	tcur = time.time()
	elapsed = (tcur-startup_time)
	samples_done = j+1
	samples_per_second = float(j+1) / elapsed
	samples_left = scene.samples_per_pixel - samples_done
	eta = samples_left / samples_per_second
	print '%d/%d,'%(samples_done,scene.samples_per_pixel), "depth: %d,"%k,
	print "s/sample: %.3f," % (tcur-t0),
	print "elapsed: %.2f s," % (tcur-startup_time),
	print "eta: %.1f min" % (eta/60.0)
	
	acc.finish()
	
	if j % itr_per_refresh == 0 or j==scene.samples_per_pixel-1:
		imgdata = img.get().astype(np.float32)[...,0:3]
		
		image.show( imgdata )
		image.save_raw( raw_output_file, imgdata )
		image.save_png( png_output_file, imgdata )
		
		acc.output_profiling_info()
	
acc.output_profiling_info()
