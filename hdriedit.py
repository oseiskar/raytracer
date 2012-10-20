
import sys, numpy, pygame, time
from scipy.misc import toimage

if len(sys.argv)==1:
	fn = raw_input("file name: ").strip()
else:
	fn = sys.argv[1]

data = numpy.load(fn)
pgwin = None

print "opened numpy matrix of size "+str(data.shape)
mean = data.mean()
data /= mean
print "mean value %s (now normalized to 1.0)" % mean
h,w = data.shape[:2]
print "assuming an image of size %d x %d" % (w,h)

ref = 1.0

def show_img(data,ref,do_save=True):
	imgdata = (numpy.clip(data*ref, 0, 1)*255).astype(numpy.uint8)
	
	global pgwin
	if not pgwin:
		pgwin = pygame.display.set_mode((w,h))
		pygame.display.set_caption("HDR image")
	
	pgwin.blit(pygame.surfarray.make_surface(imgdata.transpose((1,0,2))), (0,0))
	pygame.display.update()
	
	if do_save: toimage(imgdata).save('out.png')

ref = 1.0

while True:
	
	if ref == "sweep":
		Nsteps = 200
		log_min = -5
		log_max = 2
		vals = numpy.linspace(log_min,log_max,Nsteps)
		vals = numpy.exp(vals)
		
		for v in vals:
			print v
			show_img(data,v,False)
			time.sleep(0.01)
	else:
		ref = float(ref)
		show_img(data,ref,True)
		
	ref = raw_input("brightness: ").strip()

#print mat
