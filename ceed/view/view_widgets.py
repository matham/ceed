'''Viewer widgets
=====================

Widgets used on the viewer side of the controller/viewer interface.
These are displayed when the second process of the viewer is running.
'''

from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.scrollview import ScrollView
from kivy.uix.behaviors.knspace import KNSpaceBehavior
from kivy.clock import Clock

from ceed.view.controller import ViewController

__all__ = ('ViewRootFocusBehavior', )


class ViewRootFocusBehavior(FocusBehavior):
    '''Adds focus behavior to the viewer.

    Whenever a key is pressed in the second process it is passed on to the
    controller.
    '''

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        ViewController.send_keyboard_down(keycode[1], modifiers)
        return True

    def keyboard_on_key_up(self, window, keycode):
        ViewController.send_keyboard_up(keycode[1])
        return True


class ControlDisplay(KNSpaceBehavior, ScrollView):

    resize_trigger = None

    def __init__(self, **kwargs):
        super(ControlDisplay, self).__init__(**kwargs)
        self.resize_trigger = Clock.create_trigger(self.resize_callback)

    def resize_callback(self, *largs):
        float_layout = self.ids.float_layout
        scalar = self.ids.scalar
        top = -scalar.y + (float_layout.height - self.height) * self.scroll_y + self.height
        skip_y = scalar.y >= 0 or self.height >= float_layout.height

        left = -scalar.x + (float_layout.width - self.width) * self.scroll_x
        skip_x = scalar.x >= 0 or self.width >= float_layout.width

        scalar.parent.size = scalar.bbox[1]
        scalar.pos = 0, 0

        if not skip_x:
            self.scroll_x = left / (float_layout.width - self.width)
        if not skip_y:
            self.scroll_y = (top - self.height) / (float_layout.height - self.height)
