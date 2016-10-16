
# ------- Image input/output utils

import numpy as np
#from graphics import PygameGraphics as Graphics # Uncomment for pygame
from graphics import PySdlGraphics as Graphics

class ColorEncodingSettings:
    def __init__(self):

        ## Color encoding pipeline

        # First the intensity values are shifted downwards so that this
        # reference point has zero brightness = (r+g+b)/3
        self.low_normalization = np.min

        # Then the values are scaled linearly so that this reference
        # point becomes equal to the below brightness value
        self.brightness_reference = np.mean
        self.brightness = 0.4

        # Then gamma correction x -> x^(1/gamma) is applied
        self.gamma = 1.8

        # If set, over-exposed values are blurred with a flare effect
        self.flares = False

        # Finally, all color values are clamped to range [0,1] and
        # encoded using 8-bit


class Image:

    def __init__( self, npy_filename = None, data = None, settings = None ):
        if npy_filename == None:
            self.data = data
        else:
            self.data = np.load( npy_filename )

        if settings is None: settings = ColorEncodingSettings()
        self.settings = settings

        self._graphics = None

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
        lightness = np.ravel(np.mean(imgdata, 2))

        lo_norm = self.settings.low_normalization
        if lo_norm is not None and lo_norm is not 0:
            min_l = lo_norm(lightness)
            imgdata -= min_l
            lightness -= min_l

        ref = self.settings.brightness_reference(lightness)

        imgdata = np.clip(imgdata/ref*self.settings.brightness, 0, None)
        imgdata = np.power(imgdata, 1.0/self.settings.gamma)

        if self.settings.flares:
            imgdata = flares(imgdata)

        imgdata = np.clip(imgdata, 0, 1.0)

        return (imgdata*255).astype(np.uint8)

    def show( self, imgdata = None ):
        imgdata = self._to_24bit( imgdata )

        img = imgdata.transpose((1, 0, 2))

        if not self._graphics:
            self._graphics, self._shrink = init_window_from_image(img)
        else:
            img = shrink_image(img, self._shrink)
        self._graphics.blit_3d_numpy_array(img)
        self._graphics.update()

def init_window_from_image(img):
    graphics = Graphics()

    img, shrink = shrink_to_fit_screen(img, graphics.get_screen_size())
    w, h = img.shape[:2]

    graphics.init_window(w, h, "Image 1:%d" % shrink)
    return (graphics, shrink)


def shrink_to_fit_screen(img, screen_size):

    screen_w, screen_h = screen_size

    shrink = 1
    w, h = img.shape[:2]

    while (h/shrink > screen_h - 100 or
           w/shrink > screen_w):
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
