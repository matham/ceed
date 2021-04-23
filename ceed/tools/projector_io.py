"""Projector test app
=====================

App to test the projector - MCS hardware data link (see
:class:`~ceed.storage.controller.DataSerializerBase`).

When run, it shows a screen that allows individually setting the corner pixel
bits. It also supports setting the video mode (proxy for frame rate) and
the r, g, and b projector LED state.

When run, and if the MCS system is also running and sampling the digital port
and pixel mode is ON for the projector, setting the individual bits in the app
should also set the MCS port it is connected to high/low.
"""
from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import BooleanProperty, StringProperty, ObjectProperty
from kivy.graphics.opengl import glEnable, GL_DITHER, glDisable
from kivy.graphics.texture import Texture
if __name__ == '__main__':
    from kivy.core.window import Window
try:
    from pypixxlib import _libdpx as libdpx
    from pypixxlib.propixx import PROPixx
    from pypixxlib.propixx import PROPixxCTRL
except ImportError:
    libdpx = PROPixx = PROPixxCTRL = None

__all__ = ('IOApp', )

kv = '''
BoxLayout:
    orientation: 'vertical'
    padding: '25dp'
    bits: bits
    spacing: '5dp'
    canvas:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            texture: app.bits_texture
            pos: 0, self.height - 1
            size: 1, 1
    Widget:
        size_hint_y: None
        height: '2dp'
    BoxLayout:
        spacing: '5dp'
        size_hint: None, None
        size: self.minimum_width, '25dp'
        Label:
            padding_x: '5dp'
            text: 'Video Mode'
            size_hint_x: None
            width: self.texture_size[0]
        Spinner:
            values: app.video_modes
            on_text: app.set_video_mode(self.text)
            text: 'RGB'
            size_hint_x: None
            width: '100dp'
        Label:
            padding_x: '5dp'
            text: 'LED Mode'
            size_hint_x: None
            width: self.texture_size[0]
        Spinner:
            values: list(app.led_modes.keys())
            on_text: app.set_led_mode(self.text)
            text: 'RGB'
            size_hint_x: None
            width: '100dp'
    GridLayout:
        id: bits
        cols: 8
        spacing: '5dp'
        size_hint: None, None
        size: self.minimum_size
    ToggleButton:
        text: 'Pixel mode: ON' if app.pixel_mode else 'Pixel mode: OFF'
        size_hint: None, None
        size: self.texture_size
        on_release: app.set_pixel_mode(self.state == 'down')
        padding: '5dp', '5dp'
        state: 'down'
    Widget


<BitToggleButton@ToggleButton>:
    padding: '5dp', '5dp'
    size_hint: None, None
    size: '45dp', self.texture_size[1]
    on_state: app.update_bits()
'''


class IOApp(App):
    """The app.
    """

    led_mode = StringProperty('RGB')
    """Current LED mode from :attr:`led_modes`.
    """

    video_mode = StringProperty('RGB')
    """Current video mode from :attr:`video_modes`.
    """

    led_modes = {'RGB': 0, 'GB': 1, 'RB': 2, 'B': 3, 'RG': 4, 'G': 5, 'R': 6,
                 'none': 7}
    """Valid LED modes.
    """

    video_modes = ['RGB', 'QUAD4X', 'QUAD12X']
    """Valid video modes.
    """

    screen_size = (1920, 1080)
    """Size of the screen.
    """

    bits_texture = ObjectProperty(None)
    """Corner pixel texture.
    """

    pixel_mode = BooleanProperty(False)
    """If the projector pixel-mode is ON.
    """

    def build(self):
        tex = self.bits_texture = Texture.create(size=(1, 1))
        tex.mag_filter = 'nearest'
        tex.min_filter = 'nearest'

        root = Builder.load_string(kv)
        grid = root.bits
        for i in range(23, -1, -1):
            grid.add_widget(Factory.BitToggleButton(text=str(i)))
        return root

    def on_start(self):
        glDisable(GL_DITHER)
        Window.clearcolor = (0, 0, 0, 1)
        Window.size = self.screen_size
        Window.left = 0
        Window.fullscreen = True
        self.set_led_mode(self.led_mode)
        self.set_video_mode(self.video_mode)
        self.update_bits()
        self.set_pixel_mode(True)

    def set_pixel_mode(self, state):
        if PROPixxCTRL is None:
            raise ImportError('Cannot open PROPixx library')
        self.pixel_mode = state

        ctrl = PROPixxCTRL()
        if state:
            ctrl.dout.enablePixelMode()
        else:
            ctrl.dout.disablePixelMode()
        ctrl.updateRegisterCache()
        ctrl.close()

    def set_led_mode(self, mode):
        '''Sets the projector's LED mode. ``mode`` can be one of
        led_modes.
        '''
        if libdpx is None:
            raise ImportError('Cannot open PROPixx library')

        libdpx.DPxOpen()
        libdpx.DPxSelectDevice('PROPixx')
        libdpx.DPxSetPPxLedMask(self.led_modes[mode])
        libdpx.DPxUpdateRegCache()
        libdpx.DPxClose()

        self.led_mode = mode

    def set_video_mode(self, mode):
        '''Sets the projector's video mode. ``mode`` can be one of
        video_modes.
        '''
        if PROPixx is None:
            raise ImportError('Cannot open PROPixx library')

        modes = {'RGB': 'RGB 120Hz', 'QUAD4X': 'RGB Quad 480Hz',
                 'QUAD12X': 'GREY Quad 1440Hz'}
        self.video_mode = mode
        dev = PROPixx()
        dev.setDlpSequencerProgram(modes[mode])
        dev.updateRegisterCache()
        dev.close()

    def update_bits(self):
        value = 0
        for i, button in enumerate(self.root.bits.children):
            if button.state == 'down':
                value |= 1 << i
        r, g, b = value & 0xFF, (value & 0xFF00) >> 8, \
            (value & 0xFF0000) >> 16
        self.bits_texture.blit_buffer(
            bytes([r, g, b]), colorfmt='rgb', bufferfmt='ubyte')


if __name__ == '__main__':
    app = IOApp()
    try:
        app.run()
    finally:
        app.bits_texture.blit_buffer(
            b'000', colorfmt='rgb', bufferfmt='ubyte')
        app.set_pixel_mode(False)
        app.set_led_mode('RGB')
        app.set_video_mode('RGB')
