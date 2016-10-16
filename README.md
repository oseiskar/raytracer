An OpenCL raytracer
===================

Usage
-------------
	python clray.py scenes/scene-dev.py

(see `python clray.py -h`)

Requirements
-------------
 * Python (2.7)
 * pyopencl (>= 2012), tested with 2012.1, 2013.2, 2014.1, 2016.2
 * Mako (for pyopencl)
 * jinja2
 * numpy, scipy
 * sympy (for implicit surfaces)
 * for visualization: sdl2 (default, needs `libsdl2-dev`) or pygame.
   To use Pygame, enable it in `imgutils.py`

![showcase](http://i.imgur.com/FWbXG91.png "Example output")

Installation
------------

See https://wiki.tiker.net/PyOpenCL/Installation for a more comprehensive
guide on installing OpenCL and PyOpenCL.

Notice that GPU-accelerated libraries almost never work out-of-the-box.
The recommended steps and the encountered problems depend on your OS version,
hardware configuration and the yearly planetary aligment. My astrological tip
for CUDA on Linux is to avoid the Debian-packaged drivers and to download the
evil proprietary ones from the [Nvidia website](http://www.nvidia.com/object/unix.html),
which, however, has a high risk of breaking any graphical desktop environment
you might be using.

Here is an example how to get AMD's CPU version of OpenCL running on Debian
Jessie in October 2016 (less likely to break things):

 1. Install OpenCL like this

        sudo aptitude install amd-libopencl1 amd-opencl-icd opencl-headers amd-opencl-dev

 2. Then install PyOpenCL, Mako etc. with pip (which itself is [often broken](http://stackoverflow.com/questions/39882200/pip-error-after-upgrading-pip-scrapy-by-pip-install-upgrade/40056431#40056431))

        export LC_ALL=C # can fix problems with pip
        sudo pip install pyopencl Mako sympy jinja2
        sudo pip install sdl2 # if using PySDL2 for default visualization

 3. Install numpy, scipy and SDL (or Pygame) using the package manager

        sudo aptitude install python-numpy python-scipy
        sudo aptitude install libsdl2-dev # for default PySDL2 visualizations
        sudo aptitude install python-pygame # for pygame visualizations
