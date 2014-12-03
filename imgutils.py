
# ------- Image input/output utils

import os
import numpy as np

class Image:
	def __init__( self, npy_filename = None ):
		if npy_filename == None: self.data = None
		else: self.data = np.load( npy_filename )
		
		self.gamma = 1.8
		self.brightness = 0.3
		self._pgwin = None
		self._shrink = 1
	
	def save_raw( self, filename, imgdata = None ):
		np.save(filename, self._sum(imgdata) )
	
	def save_png( self, filename, imgdata = None ):
		from scipy.misc import toimage
		toimage(self._to_24bit(imgdata)).save(filename)
	
	def _sum( self, imgdata ):
		if self.data != None: imgdata = imgdata + self.data
		return imgdata
	
	def _to_24bit( self, imgdata = None ):
		imgdata = self._sum( imgdata )
		ref = np.mean(imgdata)
		imgdata = np.clip(imgdata/ref*self.brightness, 0, 1)
		imgdata = np.power(imgdata, 1.0/self.gamma)
		return (imgdata*255).astype(np.uint8)
	
	def show( self, imgdata ):
		imgdata = self._to_24bit( imgdata )
		import pygame
		
		h,w = imgdata.shape[:2]
		
		if not self._pgwin:
			pygame.display.init()
			screen_info = pygame.display.Info()
			self._shrink = 1
			
			while (h/self._shrink > screen_info.current_h or
			       w/self._shrink > screen_info.current_w):
				self._shrink += 1
		
			self._pgwin = pygame.display.set_mode((w/self._shrink,h/self._shrink))
			pygame.display.set_caption("Image 1:%d" % self._shrink)
		
		displayed_img = imgdata.transpose((1,0,2))[::self._shrink,::self._shrink,:]
		self._pgwin.blit(pygame.surfarray.make_surface(displayed_img), (0,0))
		pygame.display.update()
