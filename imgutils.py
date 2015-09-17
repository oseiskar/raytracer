
# ------- Image input/output utils

import numpy as np

class Image:
    def __init__( self, npy_filename = None, data = None ):
        if npy_filename == None:
            self.data = data
        else:
            self.data = np.load( npy_filename )
        
        self.gamma = 1.8
        self.brightness = 0.3
        self.normalization = 'mean'
        self.equalize = True
        self.flares = False
        self._pgwin = None
    
    def save_raw( self, filename, imgdata = None ):
        np.save(filename, self._sum(imgdata) )
    
    def save_png( self, filename, imgdata = None ):
        from scipy.misc import toimage
        toimage(self._to_24bit(imgdata)).save(filename)
    
    def _sum( self, imgdata ):
        if self.data is not None:
            if imgdata is None:
                imgdata = self.data*1
            else:
                imgdata = imgdata + self.data
        return imgdata
    
    def _to_24bit( self, imgdata = None ):
        imgdata = np.nan_to_num(self._sum( imgdata ))
        ref = getattr(np, self.normalization)(imgdata)
        
        imgdata = np.clip(imgdata/ref*self.brightness, 0, None)
        imgdata = np.power(imgdata, 1.0/self.gamma)
        if self.flares:
            imgdata = flares(imgdata)
        imgdata = np.clip(imgdata, 0, 1.0)
        if self.equalize:
            lightness = np.mean(imgdata, 2)
            min_l = np.min(lightness)
            max_l = np.max(lightness)
            
            if min_l != max_l:
                imgdata = (imgdata - min_l) / (max_l - min_l)
                imgdata = np.clip(imgdata, 0, 1.0)
        
        return (imgdata*255).astype(np.uint8)
    
    def show( self, imgdata = None ):
        imgdata = self._to_24bit( imgdata )
        
        img = imgdata.transpose((1, 0, 2))
        
        if not self._pgwin:
            self._pgwin, self._shrink = init_pygame_window_from_image(img)
        else:
            img = shrink_image(img, self._shrink)
            update_pygame_window_to_image(self._pgwin, img)
        
def init_pygame_window_from_image(img):
    import pygame
    pygame.display.init()
    
    img, shrink = shrink_to_fit_screen(img)
    w, h = img.shape[:2]

    pgwin = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Image 1:%d" % shrink)
    
    update_pygame_window_to_image(pgwin, img)
    
    return (pgwin, shrink)
    
def update_pygame_window_to_image(pgwin, img):
    import pygame
    pgwin.blit(pygame.surfarray.make_surface(img), (0, 0))
    pygame.display.update()
    
def shrink_to_fit_screen(img):
    import pygame
    screen_info = pygame.display.Info()
    shrink = 1
    w, h = img.shape[:2]
    
    while (h/shrink > screen_info.current_h * 0.7 or
           w/shrink > screen_info.current_w):
        print h/shrink, screen_info.current_h * 0.7
        shrink += 1

    img = shrink_image(img, shrink)
    return (img, shrink)

def shrink_image(img, shrink_ratio):
    return img[::shrink_ratio, ::shrink_ratio, :]

def flares(imgdata):
    """Flare effect for overexposure"""
    import scipy
    
    visiblerange = np.clip(imgdata, 0, 1)
    overexposure = imgdata - visiblerange
    
    sigma = 1.0
    
    for c in xrange(3):
        overexposure[:, :, c] = \
          scipy.ndimage.filters.gaussian_filter(overexposure[:, :, c], sigma)
          
    imgdata = visiblerange + overexposure
    
    visiblerange = np.clip(imgdata, 0, 2)
    overexposure = imgdata - visiblerange
    
    l = 100
    kernel = np.arange(0, l, dtype=np.float32)
    kernel = np.exp(-kernel * 0.2)
    kernel = np.concatenate((kernel[-1:1:-1], kernel))
    kernel /= kernel.sum()
    
    overexposure = scipy.ndimage.filters.convolve1d(overexposure, kernel, 0, None, 'constant', 0, 0)
    
    imgdata = visiblerange + overexposure
    return imgdata
