
import sys, numpy, pygame, time, scipy
import scipy.ndimage
from imgutils import Image
from scipy.misc import toimage

# pylint: disable-msg=C0103

if len(sys.argv)==1:
    fns = [raw_input("file name: ").strip()]
else:
    fns = sys.argv[1:]

data = None
for fn in fns:
    d = numpy.nan_to_num(numpy.load(fn))

    print "opened numpy matrix of size "+str(d.shape)
    
    if data == None:
        data = d
    else:
        data += d

image = Image(data = data)
image.show()

print """-----------
Valid commans are:
  [any number]
    (for example 0.1, 2, 0.9234)
    show and output an image whose brightness has been multiplied by that
    number (e.g. 2 outputs an image that is "two times brighter" than the image
    corresponding to 1)
    
  gamma [number] (or g [number])
    set gamma correction value (default %s)
    
  sweep (or s)
    display the image using various brightnesses

  flares (or f)
    toggle flares (affect over-exposure)

  CTRL-D (or any unrecognized input)
    quit / crash :)

The first and second of the above output an image named 'out-24bit.png'
-----------
""" % image.gamma

while True:
    
    line = raw_input("cmd: ").split()
    cmd = line[0]
    
    if cmd == "sweep" or cmd == "s":
        
        bright0 = image.brightness
        
        Nsteps = 200
        log_min = -5
        log_max = 2
        vals = numpy.linspace(log_min, log_max, Nsteps)
        vals = numpy.exp(vals)
        
        for v in vals:
            print v
            image.brightness = v
            image.show(data)
            time.sleep(0.01)
        
        image.brightness = bright0
    else:
        if cmd == "flares" or cmd == "f":
            image.flares = image.flares == False
            print "flares %s" % image.flares
        elif cmd == "gamma" or cmd == "g":
            image.gamma = float(line[1])
        elif cmd == "brightness" or cmd == "b":
            image.brightness = float(line[1])
        else:
            image.brightness = float(cmd)
        image.show(data)
        image.save_png('hdriedit-out.png')
