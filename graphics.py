"""
Wrappers for Pygame and PySDL2 for displaying graphics
"""

class PySdlGraphics:
    def __init__(self):
        import sdl2.ext
        sdl2.ext.init()

    def init_window(self, width, height, caption=""):
        import sdl2.ext
        self._window = sdl2.ext.Window(caption, size=(width, height))
        self._window.show()

    def update(self):
        self._window.refresh()

    def blit_3d_numpy_array(self, img):
        import sdl2.ext
        view = sdl2.ext.pixels3d(self._window.get_surface())
        view[:,:,:3] = img[:,:,::-1]

    def get_screen_size(self):
        import sdl2
        dm = sdl2.SDL_DisplayMode()
        assert(sdl2.SDL_GetCurrentDisplayMode(0, dm) == 0)
        return (dm.w, dm.h)

class PygameGraphics:
    def __init__(self):
        import pygame
        pygame.display.init()

    def init_window(self, width, height, caption=None):
        import pygame
        self._window = pygame.display.set_mode((width, height))
        if caption is not None:
            pygame.display.set_caption(caption)

    def update(self):
        import pygame
        pygame.display.update()

    def blit_3d_numpy_array(self, img):
        import pygame
        self._window.blit(pygame.surfarray.make_surface(img), (0, 0))

    def get_screen_size(self):
        import pygame
        screen_info = pygame.display.Info()
        return (screen_info.current_w, screen_info.current_w)
