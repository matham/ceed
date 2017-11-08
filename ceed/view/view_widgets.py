'''Viewer widgets
=====================

Widgets used on the viewer side of the controller/viewer interface.
These are displayed when the second process of the viewer is running.
'''

from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import Scatter
from kivy.clock import Clock

from ceed.view.controller import ViewController

__all__ = ('ViewRootFocusBehavior', )


class ViewRootFocusBehavior(FocusBehavior):
    '''Adds focus behavior to the viewer.

    Whenever a key is pressed in the second process it is passed on to the
    controller.
    '''

    _ctrl_down = False

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = True
        ViewController.send_keyboard_down(keycode[1], modifiers)
        return True

    def keyboard_on_key_up(self, window, keycode):
        if keycode[1] in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = False

        if self._ctrl_down:
            if keycode[1] == 'q':
                ViewController.filter_background = not ViewController.filter_background
                return True
        ViewController.send_keyboard_up(keycode[1])
        return True

    def keyboard_on_textinput(self, window, text):
        if not self._ctrl_down:
            return True

        if text == '+':
            ViewController.alpha_color = min(1., ViewController.alpha_color + .01)
        elif text == '-':
            ViewController.alpha_color = max(0., ViewController.alpha_color - .01)
        return True


class ControlDisplay(StencilView):

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(ControlDisplay, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(ControlDisplay, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(ControlDisplay, self).on_touch_up(touch)


class PainterScatter(Scatter):

    _sizing_trigger = None

    _pos_trigger = None

    def __init__(self, **kwargs):
        super(PainterScatter, self).__init__(**kwargs)
        self._sizing_trigger = Clock.create_trigger(self._recalculate_size, -1)
        self.fbind('scale', self._sizing_trigger)
        ViewController.fbind('screen_height', self._sizing_trigger)
        ViewController.fbind('screen_width', self._sizing_trigger)

        self._pos_trigger = Clock.create_trigger(self._recalculate_pos, -1)
        self.fbind('pos', self._pos_trigger)
        self.fbind('bbox', self._pos_trigger)

    def _recalculate_size(self, *largs):
        parent = self.parent
        self.scale = max(
            self.scale, min(1, min(parent.height / ViewController.screen_height,
                                   parent.width / ViewController.screen_width)))

    def _recalculate_pos(self, *largs):
        parent = self.parent
        x = min(max(self.x, parent.right - self.bbox[1][0]), parent.x)
        y = min(max(self.y, parent.top - self.bbox[1][1]), parent.y)
        self.pos = x, y
