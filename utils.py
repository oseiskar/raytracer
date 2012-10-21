
import numpy as np

def normalize(vecs):
	lens = np.sum(vecs**2, len(vecs.shape)-1)
	lens = np.sqrt(lens)
	lens = np.array(lens)
	lens.shape += (1,)
	return vecs / lens

def camera_rotmat(direction, up=(0,0,1)):
	
	# TODO camera roll
	
	direction = np.array(direction)
	up = np.array(up)
	
	right = np.cross(direction,up)
	up = np.cross(right,direction)
	
	rotmat = np.vstack((right,-up,direction))
	
	return normalize(rotmat).transpose()

def camera_rays(wh, flat=False, wfov=60, direction=(0,1,0), up=(0,0,1)):
	
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
