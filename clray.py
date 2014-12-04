import numpy as np
import time, sys, os, os.path, argparse
#import objgraph

from accelerator import Accelerator
from utils import *
from imgutils import Image
import shader

startup_time = time.time()

PNG_OUTPUT_FILE = 'out.png'
RAW_OUTPUT_FILE = 'out.raw.npy'

# ------- Parse options

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-a', '--append', action='store_true')
arg_parser.add_argument('-i', '--interactive_opencl_context', action='store_true')
arg_parser.add_argument('-itr', '--itr_per_refresh', type=int, default=100)
arg_parser.add_argument('scene')
args = arg_parser.parse_args()

# ------- Import scene (not pretty...)
def import_scene():
	sys.path.append('scenes/')
	scene_name = os.path.basename(args.scene).split('.')[0]
	scene_module = __import__(scene_name)
	return scene_module.scene
scene = import_scene()

# ------------- Initialize image

def init_image():
	if args.append: old_raw_file = RAW_OUTPUT_FILE
	else: old_raw_file = None
	image = Image( old_raw_file )
	image.gamma = scene.gamma
	image.brightness = scene.brightness
	return image
image = init_image()

# ------------- set up camera

cam = scene.get_camera_rays()
rotmat = scene.get_camera_rotmat()
fovx_rad = scene.camera_fov / 180.0 * np.pi
pixel_angle = fovx_rad / scene.image_size[0]

# ------------- Initialize CL

acc = Accelerator(cam.size / 3, args.interactive_opencl_context)
prog = acc.build_program( shader.make_program(scene) )

# ------------- Find root container object
Nobjects = len(scene.objects)
root_object_id = 0
for i in range(Nobjects):
	if scene.root_object == scene.objects[i]:
		root_object_id = i+1

# ------------- Parameter arrays

# Materials
fog = False
dispersion = False

object_materials = [obj.material for obj in scene.objects]
def new_mat_buf(pname):
	
	global fog, dispersion
	
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
	
	if pname == 'dispersion' and buf.sum() != 0:
		dispersion = True
		print 'dispersion'
		
	return acc.new_const_buffer(buf)

mat_diffuse = new_mat_buf('diffuse')
mat_emission = new_mat_buf('emission')
mat_reflection = new_mat_buf('reflection')
mat_transparency = new_mat_buf('transparency')
mat_vs = new_mat_buf('vs')
mat_ior = new_mat_buf('ior')
mat_dispersion = new_mat_buf('dispersion')
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
if dispersion: raycolor = acc.new_array(imgshape, np.float32, True )
else: raycolor = acc.zeros_like(img)
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
		
		if dispersion:
			
			wavelength = np.float32(np.random.rand())
			def tent_func(x):
				if abs(x) > 1.0: return 0
				return 1.0 - abs(x)
			
			color_mask = [tent_func( 4.0 * (wavelength-c) ) for c in [0.25,0.5,0.75]]
			
			hostbuf[1,:3] = color_mask
			dispersion_coeff = np.float32((wavelength - 0.5) * 2.0)
			
			#print wavelength, color_mask, dispersion_coeff
		
		#hostbuf[2,0] = light.R
		acc.enqueue_copy(vec_broadcast, hostbuf)
		
		if dispersion:
			acc.call('prob_select_ray_dispersive', \
				(img, whichobject, normal,isec_dist,pos,ray,raycolor,inside), \
				(mat_emission, mat_diffuse,mat_reflection,mat_transparency,
				mat_ior,mat_dispersion,mat_vs, rand_01,dispersion_coeff,
				vec_broadcast))
		else:
			acc.call('prob_select_ray', \
				(img, whichobject, normal,isec_dist,pos,ray,raycolor,inside), \
				(mat_emission, mat_diffuse,mat_reflection,mat_transparency,
				 mat_ior,mat_vs,rand_01,vec_broadcast))
		
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
	
	if j % args.itr_per_refresh == 0 or j==scene.samples_per_pixel-1:
		imgdata = img.get().astype(np.float32)[...,0:3]
		
		image.show( imgdata )
		image.save_raw( RAW_OUTPUT_FILE, imgdata )
		image.save_png( PNG_OUTPUT_FILE, imgdata )
		
		acc.output_profiling_info()
	
acc.output_profiling_info()
