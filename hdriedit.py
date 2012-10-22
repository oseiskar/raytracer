
import sys, numpy, pygame, time, scipy
import scipy.ndimage
from scipy.misc import toimage

if len(sys.argv)==1:
	fns = [raw_input("file name: ").strip()]
else:
	fns = sys.argv[1:]

data = None
for fn in fns:
	d = numpy.load(fn)

	print "opened numpy matrix of size "+str(d.shape)
	
	if data == None: data = d
	else: data += d

mean = data.mean()
data /= mean
print "mean value %s (now normalized to 1.0)" % mean
h,w = data.shape[:2]
print "assuming an image of size %d x %d" % (w,h)
	
ref = 1.0
gamma = 1.8
pgwin = None

def flares(imgdata):
	visiblerange = numpy.clip(imgdata,0,1)
	overexposure = imgdata - visiblerange
	
	sigma = 1.0
	
	for c in xrange(3):
		overexposure[:,:,c] = \
		  scipy.ndimage.filters.gaussian_filter(overexposure[:,:,c], sigma)
		  
	imgdata = visiblerange + overexposure
	
	visiblerange = numpy.clip(imgdata,0,2)
	overexposure = imgdata - visiblerange
	
	l = 100
	kernel = numpy.arange(0,l, dtype=numpy.float32)
	kernel = numpy.exp(-kernel * 0.2)
	kernel = numpy.concatenate((kernel[-1:1:-1],kernel))
	kernel /= kernel.sum()
	
	overexposure = scipy.ndimage.filters.convolve1d(overexposure, kernel, 0, None, 'constant', 0, 0)
	
	imgdata = visiblerange + overexposure
	return imgdata

def show_img(data,ref,do_flares=False,do_save=True):
	
	imgdata = data*ref
	imgdata = numpy.power(imgdata, 1.0/gamma)
	if do_flares: imgdata = flares(imgdata)
	
	imgdata = (numpy.clip(imgdata, 0, 1)*255).astype(numpy.uint8)
	
	global pgwin
	if not pgwin:
		pgwin = pygame.display.set_mode((w,h))
		pygame.display.set_caption("HDR image")
	
	pgwin.blit(pygame.surfarray.make_surface(imgdata.transpose((1,0,2))), (0,0))
	pygame.display.update()
	
	if do_save: toimage(imgdata).save('out-24bit.png')

ref = 1.0
do_flares = False
show_img(data,ref,do_flares,False)

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
""" % gamma

while True:
	
	line = raw_input("cmd: ").split()
	cmd = line[0]
	
	if cmd == "sweep" or cmd == "s":
		Nsteps = 200
		log_min = -5
		log_max = 2
		vals = numpy.linspace(log_min,log_max,Nsteps)
		vals = numpy.exp(vals)
		
		for v in vals:
			print v
			show_img(data,v,False,False)
			time.sleep(0.01)
	else:
		if cmd == "flares" or cmd == "f":
			do_flares = do_flares == False
			print "flares %s" % do_flares
		elif cmd == "gamma" or cmd == "g":
			gamma = float(line[1])
		elif cmd == "brightness" or cmd == "b":
			ref = float(line[1])
		else:
			ref = float(cmd)
		show_img(data,ref,do_flares,True)

#print mat
