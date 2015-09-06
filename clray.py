"""
Program entry point.

To use, an instance of the Scene class should be defined in a scene file (e.g.,
scenes/scene-test.py), which is provided to this program as a command
line argument.

The rendering process is facilitated by the Renderer class, which generates, 
compiles and runs OpenCL code that renders the given scene. The resulting
image and the intermediate results are written to PNG_OUTPUT_FILE and
RAW_OUTPUT_FILE (as well as displayed on a preview window, unless otherwise
specified by the command line arguments).
"""

if __name__ == '__main__':
    
    import numpy as np
    import time, sys, os, os.path, argparse
    import renderer

    from imgutils import Image

    startup_time = time.time()

    PNG_OUTPUT_FILE = 'out.png'
    RAW_OUTPUT_FILE = 'out.raw.npy'

    # ------- Parse options

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--append', action='store_true')
    arg_parser.add_argument('-n', '--no_window', action='store_true')
    arg_parser.add_argument('-itr', '--itr_per_refresh', type=int, default=100)
    arg_parser.add_argument('-o', '--cl_build_options', default='')
    arg_parser.add_argument('-c', '--choose_opencl_context', \
        action='store_true')
    
    arg_parser.add_argument('scene')
    args = arg_parser.parse_args()

    # ------- Import scene (not pretty...)
    def import_scene():
        sys.path.append(os.path.dirname(args.scene))
        scene_name = os.path.basename(args.scene).split('.')[0]
        scene_module = __import__(scene_name)
        return scene_module.scene
    scene = import_scene()

    # ------------- Initialize image

    def init_image():
        if args.append:
            old_raw_file = RAW_OUTPUT_FILE
        else: old_raw_file = None
        img = Image( old_raw_file )
        img.gamma = scene.gamma
        img.brightness = scene.brightness
        img.normalization = scene.brightness_reference
        return img
    image = init_image()

    # ------------- Initialize CL

    renderer = renderer.Renderer(scene, args)
    acc = renderer.acc

    # Do it
    for j in xrange(scene.samples_per_pixel):
        
        t0 = time.time()
        depth = renderer.render_sample(j)
        tcur = time.time()
        
        elapsed = (tcur-startup_time)
        samples_done = j+1
        samples_per_second = float(j+1) / elapsed
        samples_left = scene.samples_per_pixel - samples_done
        eta = samples_left / samples_per_second
        rays_per_second = int(renderer.rays_per_sample() * samples_per_second)
        
        print '%d/%d,' % (samples_done, scene.samples_per_pixel), \
            "depth: %d," % depth,
        
        #print "s/sample: %.3f," % (tcur-t0),
        print '%d rays/s' % rays_per_second,
        print "elapsed: %.2f s," % (tcur-startup_time),
        print "eta: %.1f min" % (eta/60.0)
        
        if j % args.itr_per_refresh == 0 or j == scene.samples_per_pixel-1 or \
           (j % max(1,int(args.itr_per_refresh/10)) == 0 and \
            j < args.itr_per_refresh):
            
            imgdata = renderer.get_image()
            print 'image mean:', np.mean(np.ravel(imgdata))
            
            if not args.no_window:
                image.show( imgdata )
            
            image.save_raw( RAW_OUTPUT_FILE, imgdata )
            image.save_png( PNG_OUTPUT_FILE, imgdata )
            
            acc.output_profiling_info()
        
    acc.output_profiling_info()
