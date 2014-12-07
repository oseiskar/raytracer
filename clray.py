import numpy as np
import time, sys, os, os.path, argparse
#import objgraph

from imgutils import Image
from shader import Shader

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

# ------------- Initialize CL

shader = scene.shader(scene, args)
acc = shader.acc

# Do it
for j in xrange(scene.samples_per_pixel):
    
    t0 = time.time()
    depth = shader.render_sample(j)
    tcur = time.time()
    
    elapsed = (tcur-startup_time)
    samples_done = j+1
    samples_per_second = float(j+1) / elapsed
    samples_left = scene.samples_per_pixel - samples_done
    eta = samples_left / samples_per_second
    
    print '%d/%d,'%(samples_done,scene.samples_per_pixel), "depth: %d,"%depth,
    print "s/sample: %.3f," % (tcur-t0),
    print "elapsed: %.2f s," % (tcur-startup_time),
    print "eta: %.1f min" % (eta/60.0)
    
    if j % args.itr_per_refresh == 0 or j==scene.samples_per_pixel-1:
        imgdata = shader.img.get().astype(np.float32)[...,0:3]
        
        image.show( imgdata )
        image.save_raw( RAW_OUTPUT_FILE, imgdata )
        image.save_png( PNG_OUTPUT_FILE, imgdata )
        
        acc.output_profiling_info()
    
acc.output_profiling_info()
